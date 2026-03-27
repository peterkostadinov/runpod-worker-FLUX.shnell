import base64
import gc
import io
import os
import warnings

# CLIP has a hard 77-token limit; FLUX relies on T5 for long prompts — suppress the noise.
warnings.filterwarnings(
    "ignore",
    message=".*CLIP can only handle sequences up to 77 tokens.*",
)

import runpod
import torch
from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, FluxPipeline, FluxTransformer2DModel
from optimum.quanto import QuantoConfig, freeze
from PIL import Image
from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate
from transformers import T5EncoderModel

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


def _decode_base64_image(b64_string):
    """Decode a base64-encoded image string to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")


class ModelHandler:
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.load_models()

    def load_models(self):
        model_id = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
        print(f"[ModelHandler] Loading model: {model_id}", flush=True)

        # Load, quantize layer-by-layer during loading (avoids holding bfloat16 + quantized
        # copies simultaneously), then push to GPU. Peak CPU RAM ~12 GB instead of ~36 GB.
        print("[ModelHandler] Loading transformer and quantizing to int8...", flush=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            quantization_config=QuantoConfig(weights="int8"),
        )
        freeze(transformer)
        transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

        # Now load T5 — transformer is already on GPU so CPU RAM is free
        print("[ModelHandler] Loading T5 text encoder and quantizing to int8...", flush=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            quantization_config=QuantoConfig(weights="int8"),
        )
        freeze(text_encoder_2)

        # Assemble txt2img pipeline with pre-quantized components
        print("[ModelHandler] Assembling pipelines...", flush=True)
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

        # Create img2img and inpaint pipelines sharing all weights (zero extra VRAM)
        self.img2img_pipe = FluxImg2ImgPipeline(**self.pipe.components)
        self.inpaint_pipe = FluxInpaintPipeline(**self.pipe.components)
        print("[ModelHandler] All pipelines ready", flush=True)


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using FLUX.1-schnell Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import json
    import pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Create generator with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    try:
        mode = job_input.get("mode", "txt2img")
        common_kwargs = dict(
            prompt=job_input["prompt"],
            height=job_input["height"],
            width=job_input["width"],
            num_inference_steps=job_input["num_inference_steps"],
            guidance_scale=job_input["guidance_scale"],
            num_images_per_prompt=job_input["num_images"],
            max_sequence_length=256,
            generator=generator,
        )

        with torch.inference_mode():
            if mode == "img2img":
                if not job_input.get("image"):
                    return {"error": "img2img mode requires an 'image' field"}
                init_image = _decode_base64_image(job_input["image"])
                output = MODELS.img2img_pipe(
                    image=init_image,
                    strength=job_input["strength"],
                    **common_kwargs,
                ).images

            elif mode == "inpainting":
                if not job_input.get("image"):
                    return {"error": "inpainting mode requires an 'image' field"}
                if not job_input.get("mask_image"):
                    return {"error": "inpainting mode requires a 'mask_image' field"}
                init_image = _decode_base64_image(job_input["image"])
                mask = _decode_base64_image(job_input["mask_image"])
                output = MODELS.inpaint_pipe(
                    image=init_image,
                    mask_image=mask,
                    strength=job_input["strength"],
                    **common_kwargs,
                ).images

            else:
                output = MODELS.pipe(**common_kwargs).images

    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True,
        }
    except Exception as err:
        print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    return results


runpod.serverless.start({"handler": generate_image})

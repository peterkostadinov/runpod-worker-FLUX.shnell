"""FLUX.1-schnell — Redis-backed GPU inference worker.

Polls `flux:jobs:queue` via BLPOP, runs inference, writes results back.
Shares the same hf_cache volume as `handler.py` so quantized weights are
reused and do not have to be re-computed between restarts.

Start locally (outside Docker):
    REDIS_URL=redis://localhost:6379 python redis_worker.py

With Docker Compose:
    docker compose up worker
"""

import base64
import gc
import io
import json
import os
import time
import warnings

import safetensors.torch

warnings.filterwarnings(
    "ignore",
    message=".*CLIP can only handle sequences up to 77 tokens.*",
)

import redis
import torch
from accelerate import init_empty_weights
from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, FluxPipeline, FluxTransformer2DModel
from dotenv import load_dotenv
from optimum.quanto import freeze, quantization_map, qint8, quantize, requantize
from PIL import Image
from transformers import T5Config, T5EncoderModel

load_dotenv()

torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
QUEUE_KEY = "flux:jobs:queue"

# Shared with handler.py so quantized weights survive container restarts.
_QUANTO_CACHE = "/root/.cache/huggingface/quanto_cache"


# ---------------------------------------------------------------------------
# Quantization cache helpers  (identical to handler.py so both share weights)
# ---------------------------------------------------------------------------

def _cache_valid(path: str, weights_file: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "quanto_qmap.json"))
        and (
            os.path.isfile(os.path.join(path, weights_file))
            or os.path.isfile(os.path.join(path, weights_file + ".index.json"))
        )
    )


def _load_safetensors(directory: str, weights_file: str) -> dict:
    single = os.path.join(directory, weights_file)
    if os.path.isfile(single):
        return safetensors.torch.load_file(single, device="cpu")
    index_path = single + ".index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    state_dict: dict = {}
    for shard in set(weight_map.values()):
        state_dict.update(safetensors.torch.load_file(os.path.join(directory, shard), device="cpu"))
    return state_dict


def _save_cache(model, cache_dir: str, **save_kwargs) -> bool:
    import shutil
    try:
        os.makedirs(cache_dir, exist_ok=True)
        model.save_pretrained(cache_dir, **save_kwargs)
        qmap = quantization_map(model)
        with open(os.path.join(cache_dir, "quanto_qmap.json"), "w", encoding="utf-8") as f:
            json.dump(qmap, f, indent=4)
        return True
    except Exception as exc:
        print(f"[worker] Warning: cache save failed ({exc}); removing partial cache", flush=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Model loading  (same quantisation strategy as handler.py)
# ---------------------------------------------------------------------------

def load_models():
    model_id = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
    print(f"[worker] Loading model: {model_id}", flush=True)

    model_slug = model_id.replace("/", "--")
    transformer_cache = os.path.join(_QUANTO_CACHE, model_slug, "transformer")
    t5_cache = os.path.join(_QUANTO_CACHE, model_slug, "text_encoder_2")

    xf_weights = "diffusion_pytorch_model.safetensors"
    t5_weights = "model.safetensors"

    if _cache_valid(transformer_cache, xf_weights) and _cache_valid(t5_cache, t5_weights):
        print("[worker] Loading pre-quantized transformer from cache...", flush=True)
        cfg = FluxTransformer2DModel.load_config(transformer_cache)
        with init_empty_weights():
            transformer = FluxTransformer2DModel.from_config(cfg)
        with open(os.path.join(transformer_cache, "quanto_qmap.json"), encoding="utf-8") as f:
            qmap = json.load(f)
        state_dict = _load_safetensors(transformer_cache, xf_weights)
        requantize(transformer, state_dict=state_dict, quantization_map=qmap)
        del state_dict
        transformer.to(dtype=torch.bfloat16)
        transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

        print("[worker] Loading pre-quantized T5 encoder from cache...", flush=True)
        cfg = T5Config.from_pretrained(t5_cache)
        with init_empty_weights():
            text_encoder_2 = T5EncoderModel(cfg)
        with open(os.path.join(t5_cache, "quanto_qmap.json"), encoding="utf-8") as f:
            qmap = json.load(f)
        state_dict = _load_safetensors(t5_cache, t5_weights)
        requantize(text_encoder_2, state_dict=state_dict, quantization_map=qmap)
        del state_dict
        gc.collect()
        text_encoder_2.to(dtype=torch.bfloat16)
    else:
        print("[worker] Loading transformer in bfloat16...", flush=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("[worker] Quantizing transformer to int8...", flush=True)
        quantize(transformer, weights=qint8)
        freeze(transformer)
        print("[worker] Saving quantized transformer to cache...", flush=True)
        _save_cache(transformer, transformer_cache)
        transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

        print("[worker] Loading T5 text encoder in bfloat16...", flush=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("[worker] Quantizing T5 text encoder to int8...", flush=True)
        quantize(text_encoder_2, weights=qint8)
        freeze(text_encoder_2)
        print("[worker] Saving quantized T5 encoder to cache...", flush=True)
        _save_cache(text_encoder_2, t5_cache, max_shard_size="20GB")

    print("[worker] Assembling pipelines...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    img2img_pipe = FluxImg2ImgPipeline(**pipe.components)
    inpaint_pipe = FluxInpaintPipeline(**pipe.components)
    print("[worker] All pipelines ready", flush=True)
    return pipe, img2img_pipe, inpaint_pipe


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(job_input: dict, pipe, img2img_pipe, inpaint_pipe) -> dict:
    seed = job_input.get("seed") or int.from_bytes(os.urandom(2), "big")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(seed)

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

    mode = job_input.get("mode", "txt2img")
    if mode == "img2img":
        if not job_input.get("image"):
            raise ValueError("img2img mode requires an 'image' field")
        init_image = _decode_base64_image(job_input["image"])
        images = img2img_pipe(image=init_image, strength=job_input["strength"], **common_kwargs).images

    elif mode == "inpainting":
        if not job_input.get("image"):
            raise ValueError("inpainting mode requires an 'image' field")
        if not job_input.get("mask_image"):
            raise ValueError("inpainting mode requires a 'mask_image' field")
        init_image = _decode_base64_image(job_input["image"])
        mask = _decode_base64_image(job_input["mask_image"])
        images = inpaint_pipe(
            image=init_image, mask_image=mask, strength=job_input["strength"], **common_kwargs
        ).images

    else:  # txt2img
        images = pipe(**common_kwargs).images

    image_urls = [_pil_to_data_uri(img) for img in images]
    return {"images": image_urls, "image_url": image_urls[0], "seed": seed}


# ---------------------------------------------------------------------------
# Main queue loop
# ---------------------------------------------------------------------------

def main():
    rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    print(f"[worker] Connected to Redis at {REDIS_URL}", flush=True)

    pipe, img2img_pipe, inpaint_pipe = load_models()
    print("[worker] Waiting for jobs...", flush=True)

    while True:
        # BLPOP blocks until a job arrives (timeout=0 = block indefinitely).
        result = rdb.blpop(QUEUE_KEY, timeout=0)
        if result is None:
            continue
        _, job_id = result
        job_key = f"flux:job:{job_id}"

        # Skip if cancelled while waiting in the queue.
        status = rdb.hget(job_key, "status")
        if status != "IN_QUEUE":
            print(f"[worker] Skipping job {job_id} (status={status})", flush=True)
            continue

        rdb.hset(job_key, mapping={
            "status": "IN_PROGRESS",
            "started_at": str(time.time()),
        })
        print(f"[worker] Processing job {job_id}", flush=True)

        try:
            job_input = json.loads(rdb.hget(job_key, "input"))
            output = run_inference(job_input, pipe, img2img_pipe, inpaint_pipe)
            rdb.hset(job_key, mapping={
                "status": "COMPLETED",
                "output": json.dumps(output),
                "completed_at": str(time.time()),
            })
            print(f"[worker] Job {job_id} completed", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rdb.hset(job_key, mapping={
                "status": "FAILED",
                "error": str(exc),
                "completed_at": str(time.time()),
            })
            print(f"[worker] Job {job_id} FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()

"""
Local FastAPI server for FLUX.1-schnell on a single GPU (e.g. 3090 24 GB).

Start:
    python local_server.py

Endpoints:
    POST /generate   — JSON body, returns generated image(s) as base64
    GET  /health     — liveness check
"""

import base64
import gc
import io
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*CLIP can only handle sequences up to 77 tokens.*",
)

import torch
import uvicorn
from diffusers import (
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from optimum.quanto import freeze, qint8, quantize
from PIL import Image
from pydantic import BaseModel, Field
from transformers import T5EncoderModel
from typing import List, Optional

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    mode: str = Field("txt2img", pattern="^(txt2img|img2img|inpainting)$")
    prompt: str = "a photo of a cat"
    image: Optional[str] = None          # base64-encoded image for img2img / inpainting
    mask_image: Optional[str] = None     # base64-encoded mask for inpainting
    strength: float = Field(0.75, ge=0.0, le=1.0)
    height: int = Field(1024, ge=256, le=2048)
    width: int = Field(1024, ge=256, le=2048)
    seed: Optional[int] = None
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=2)


class GenerateResponse(BaseModel):
    images: List[str]
    image_url: str
    seed: int


# ---------------------------------------------------------------------------
# Helpers
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
# Model loading (same quantisation strategy as handler.py)
# ---------------------------------------------------------------------------

def load_models():
    model_id = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
    print(f"[local_server] Loading model: {model_id}", flush=True)

    print("[local_server] Loading transformer (bfloat16)…", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    print("[local_server] Quantizing transformer → int8…", flush=True)
    quantize(transformer, weights=qint8)
    freeze(transformer)
    transformer.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()

    print("[local_server] Loading T5 text encoder (bfloat16)…", flush=True)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    print("[local_server] Quantizing T5 → int8…", flush=True)
    quantize(text_encoder_2, weights=qint8)
    freeze(text_encoder_2)

    print("[local_server] Assembling pipelines…", flush=True)
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    img2img_pipe = FluxImg2ImgPipeline(**pipe.components)
    inpaint_pipe = FluxInpaintPipeline(**pipe.components)

    print("[local_server] All pipelines ready ✓", flush=True)
    return pipe, img2img_pipe, inpaint_pipe


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FLUX.1-schnell Local Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipelines are loaded once at startup
pipe: FluxPipeline = None  # type: ignore[assignment]
img2img_pipe: FluxImg2ImgPipeline = None  # type: ignore[assignment]
inpaint_pipe: FluxInpaintPipeline = None  # type: ignore[assignment]


@app.on_event("startup")
async def startup():
    global pipe, img2img_pipe, inpaint_pipe
    pipe, img2img_pipe, inpaint_pipe = load_models()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}


@app.post("/generate", response_model=GenerateResponse)
@torch.inference_mode()
def generate(req: GenerateRequest):
    seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(2), "big")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(seed)

    common_kwargs = dict(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        num_images_per_prompt=req.num_images,
        max_sequence_length=256,
        generator=generator,
    )

    if req.mode == "img2img":
        if not req.image:
            return {"error": "img2img mode requires an 'image' field"}
        init_image = _decode_base64_image(req.image)
        images = img2img_pipe(
            image=init_image, strength=req.strength, **common_kwargs
        ).images

    elif req.mode == "inpainting":
        if not req.image:
            return {"error": "inpainting mode requires an 'image' field"}
        if not req.mask_image:
            return {"error": "inpainting mode requires a 'mask_image' field"}
        init_image = _decode_base64_image(req.image)
        mask = _decode_base64_image(req.mask_image)
        images = inpaint_pipe(
            image=init_image, mask_image=mask, strength=req.strength, **common_kwargs
        ).images

    else:  # txt2img
        images = pipe(**common_kwargs).images

    image_urls = [_pil_to_data_uri(img) for img in images]

    return GenerateResponse(images=image_urls, image_url=image_urls[0], seed=seed)


# ---------------------------------------------------------------------------
# Run with:  python local_server.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

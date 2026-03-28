"""
Local FastAPI server for FLUX.1-schnell on a single GPU (e.g. 3090 24 GB).

Start:
    API_KEY=your-secret-key python local_server.py

If API_KEY is not set, a random key is generated and printed at startup.

Endpoints:
    POST /generate          — synchronous: blocks until image is ready, returns base64
    POST /run               — async: queues job, returns {id} immediately (RunPod-compatible)
    GET  /status/{job_id}   — poll job status; output is populated when COMPLETED
    POST /cancel/{job_id}   — cancel a queued job
    GET  /health            — liveness check
"""

import asyncio
import base64
import gc
import io
import os
import secrets
import time
import uuid
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

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
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from optimum.quanto import freeze, qint8, quantize
from PIL import Image
from pydantic import BaseModel, Field
from transformers import T5EncoderModel

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
# Job queue models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunRequest(BaseModel):
    """RunPod-compatible job submission wrapper."""
    input: GenerateRequest


class JobSubmitResponse(BaseModel):
    id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    id: str
    status: JobStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


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
# API key auth
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("API_KEY") or secrets.token_urlsafe(32)

security = HTTPBearer()


def _verify_key(creds: HTTPAuthorizationCredentials = Security(security)):
    if not secrets.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


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

# Job queue — populated by /run, drained by _queue_worker
_job_store: Dict[str, Dict[str, Any]] = {}
_job_queue: asyncio.Queue  # type: ignore[assignment]  — set in startup
_inference_lock: asyncio.Lock  # type: ignore[assignment]  — set in startup


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------

def _run_inference(req: GenerateRequest) -> GenerateResponse:
    """Blocking GPU inference — called from a ThreadPoolExecutor."""
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

    with torch.inference_mode():
        if req.mode == "img2img":
            if not req.image:
                raise ValueError("img2img mode requires an 'image' field")
            init_image = _decode_base64_image(req.image)
            images = img2img_pipe(
                image=init_image, strength=req.strength, **common_kwargs
            ).images

        elif req.mode == "inpainting":
            if not req.image:
                raise ValueError("inpainting mode requires an 'image' field")
            if not req.mask_image:
                raise ValueError("inpainting mode requires a 'mask_image' field")
            init_image = _decode_base64_image(req.image)
            mask = _decode_base64_image(req.mask_image)
            images = inpaint_pipe(
                image=init_image, mask_image=mask, strength=req.strength, **common_kwargs
            ).images

        else:  # txt2img
            images = pipe(**common_kwargs).images

    image_urls = [_pil_to_data_uri(img) for img in images]
    return GenerateResponse(images=image_urls, image_url=image_urls[0], seed=seed)


async def _queue_worker() -> None:
    """Drains _job_queue one job at a time; inference runs in a thread pool."""
    loop = asyncio.get_event_loop()
    while True:
        job_id: str = await _job_queue.get()
        job = _job_store.get(job_id)
        if job is None or job["status"] == JobStatus.CANCELLED:
            _job_queue.task_done()
            continue

        job["status"] = JobStatus.IN_PROGRESS
        job["started_at"] = time.time()
        try:
            async with _inference_lock:
                result: GenerateResponse = await loop.run_in_executor(
                    None, _run_inference, job["input"]
                )
            job["status"] = JobStatus.COMPLETED
            job["output"] = result.model_dump()
        except Exception as exc:
            job["status"] = JobStatus.FAILED
            job["error"] = str(exc)
            print(f"[queue_worker] Job {job_id} failed: {exc}", flush=True)
        finally:
            job["completed_at"] = time.time()
            _job_queue.task_done()


@app.on_event("startup")
async def startup():
    global pipe, img2img_pipe, inpaint_pipe, _job_queue, _inference_lock
    pipe, img2img_pipe, inpaint_pipe = load_models()
    _job_queue = asyncio.Queue()
    _inference_lock = asyncio.Lock()
    asyncio.create_task(_queue_worker())
    print(f"[local_server] API_KEY = {API_KEY}", flush=True)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, _key=Depends(_verify_key)):
    """Synchronous endpoint — holds the connection open until the image is ready."""
    loop = asyncio.get_event_loop()
    async with _inference_lock:
        return await loop.run_in_executor(None, _run_inference, req)


@app.post("/run", response_model=JobSubmitResponse, status_code=202)
async def submit_job(req: RunRequest, _key=Depends(_verify_key)):
    """Queue a job and return immediately with a job ID. Poll /status/{id} for the result."""
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "id": job_id,
        "status": JobStatus.IN_QUEUE,
        "input": req.input,
        "output": None,
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
    }
    await _job_queue.put(job_id)
    return JobSubmitResponse(id=job_id, status=JobStatus.IN_QUEUE)


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str, _key=Depends(_verify_key)):
    """Poll a job. When status is COMPLETED, output contains the generation result."""
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(
        id=job["id"],
        status=job["status"],
        output=job["output"],
        error=job["error"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
    )


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str, _key=Depends(_verify_key)):
    """Cancel a queued job. Has no effect on in-progress or completed jobs."""
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job["status"] == JobStatus.IN_QUEUE:
        job["status"] = JobStatus.CANCELLED
        job["completed_at"] = time.time()
    return {"id": job_id, "status": job["status"]}


# ---------------------------------------------------------------------------
# Run with:  python local_server.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

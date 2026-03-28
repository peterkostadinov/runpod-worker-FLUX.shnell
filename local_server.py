"""FLUX.1-schnell Job Queue API — Redis-backed gateway.

Accepts job submissions from your platform, pushes them onto a Redis queue,
and lets callers poll for results.  All model loading and inference runs in
the separate `worker` container (redis_worker.py); this process is
intentionally lightweight and has no ML dependencies.

Start locally (outside Docker):
    REDIS_URL=redis://localhost:6379 API_KEY=secret uvicorn local_server:app

With Docker Compose:
    docker compose up

Endpoints:
    POST /run               — queue a job; returns {"id": "...", "status": "IN_QUEUE"}
    GET  /status/{job_id}   — poll status; output is populated when COMPLETED
    POST /cancel/{job_id}   — cancel a queued (not yet started) job
    GET  /health            — liveness + Redis connectivity check
"""

import json
import os
import secrets
import time
import uuid
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
QUEUE_KEY = "flux:jobs:queue"
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/images")
JOB_TTL = 24 * 60 * 60  # seconds — job data auto-expires after 24 h

API_KEY = os.environ.get("API_KEY") or secrets.token_urlsafe(32)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

security = HTTPBearer()


def _verify_key(creds: HTTPAuthorizationCredentials = Security(security)):
    if not secrets.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GenerateRequest(BaseModel):
    mode: str = Field("txt2img", pattern="^(txt2img|img2img|inpainting)$")
    prompt: str = "a photo of a cat"
    image: Optional[str] = None          # base64-encoded source image (img2img / inpainting)
    mask_image: Optional[str] = None     # base64-encoded mask (inpainting only)
    strength: float = Field(0.75, ge=0.0, le=1.0)
    height: int = Field(1024, ge=256, le=2048)
    width: int = Field(1024, ge=256, le=2048)
    seed: Optional[int] = None
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=2)


class RunRequest(BaseModel):
    """RunPod-compatible job submission wrapper."""
    input: GenerateRequest


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FLUX.1-schnell Job Queue API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rdb: aioredis.Redis = None  # type: ignore[assignment]


@app.on_event("startup")
async def startup():
    global rdb
    rdb = aioredis.Redis.from_url(REDIS_URL, decode_responses=True)
    print(f"[api] Redis: {REDIS_URL}", flush=True)
    print(f"[api] API_KEY = {API_KEY}", flush=True)


@app.on_event("shutdown")
async def shutdown():
    await rdb.aclose()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        await rdb.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


@app.post("/run", status_code=202)
async def submit_job(req: RunRequest, _key=Depends(_verify_key)):
    """Queue a generation job. Returns immediately with a job ID."""
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "status": JobStatus.IN_QUEUE,
        "input": req.input.model_dump_json(),
        "output": "",
        "error": "",
        "created_at": str(time.time()),
        "started_at": "",
        "completed_at": "",
    }
    async with rdb.pipeline() as pipe:
        await pipe.hset(f"flux:job:{job_id}", mapping=job_data)
        await pipe.expire(f"flux:job:{job_id}", JOB_TTL)
        await pipe.rpush(QUEUE_KEY, job_id)
        await pipe.execute()
    return {"id": job_id, "status": JobStatus.IN_QUEUE}


@app.get("/status/{job_id}")
async def job_status(job_id: str, _key=Depends(_verify_key)):
    """Poll a job. When status is COMPLETED, output contains the generation result."""
    job = await rdb.hgetall(f"flux:job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {
        "id": job["id"],
        "status": job["status"],
        "output": json.loads(job["output"]) if job.get("output") else None,
        "error": job.get("error") or None,
        "created_at": float(job["created_at"]) if job.get("created_at") else None,
        "started_at": float(job["started_at"]) if job.get("started_at") else None,
        "completed_at": float(job["completed_at"]) if job.get("completed_at") else None,
    }


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str, _key=Depends(_verify_key)):
    """Cancel a queued job. Has no effect on in-progress or completed jobs."""
    job_key = f"flux:job:{job_id}"
    status = await rdb.hget(job_key, "status")
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if status == JobStatus.IN_QUEUE:
        await rdb.hset(job_key, mapping={
            "status": JobStatus.CANCELLED,
            "completed_at": str(time.time()),
        })
        status = JobStatus.CANCELLED
    return {"id": job_id, "status": status}


@app.get("/image/{job_id}/{index}")
async def serve_image(job_id: str, index: int, _key=Depends(_verify_key)):
    """Download a generated image file by job ID and image index."""
    path = os.path.join(IMAGE_DIR, f"{job_id}_{index}.png")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found or expired")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

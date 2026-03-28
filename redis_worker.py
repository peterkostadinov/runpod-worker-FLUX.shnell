"""FLUX.1-schnell — Redis queue bridge.

Dequeues jobs from Redis (BLPOP), forwards each one to the RunPod handler's
local HTTP API (POST /runsync on the `worker` service), saves the generated
images to the shared /images volume, writes the result back to Redis with a
60-minute TTL, and deletes image files when their Redis key expires.

No ML code here — all model loading and inference live inside handler.py.

Start with Docker Compose:
    docker compose up redis_worker
"""

import base64
import json
import os
import threading
import time

import redis
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL  = os.environ.get("REDIS_URL",  "redis://redis:6379")
WORKER_URL = os.environ.get("WORKER_URL", "http://worker:8000")
QUEUE_KEY  = "flux:jobs:queue"
IMAGE_DIR  = os.environ.get("IMAGE_DIR",  "/images")
JOB_TTL    = 60 * 60       # 60 minutes — Redis key TTL after job completes
WORKER_RETRY_DELAY = 5     # seconds between retries when worker is not yet ready
WORKER_TIMEOUT     = 300   # seconds to wait for a single /runsync call

rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def save_images(job_id: str, images: list) -> int:
    """Decode base64 data-URI images and write them to IMAGE_DIR.
    Returns the number of files saved."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    saved = 0
    for i, data_uri in enumerate(images):
        if isinstance(data_uri, str) and data_uri.startswith("data:image"):
            b64 = data_uri.split(",", 1)[1]
            path = os.path.join(IMAGE_DIR, f"{job_id}_{i}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            saved += 1
    return saved


def delete_images(job_id: str):
    """Remove all stored image files for a job."""
    deleted, i = 0, 0
    while True:
        path = os.path.join(IMAGE_DIR, f"{job_id}_{i}.png")
        if os.path.isfile(path):
            os.remove(path)
            deleted += 1
            i += 1
        else:
            break
    if deleted:
        print(f"[redis_worker] Deleted {deleted} image(s) for expired job {job_id}", flush=True)


# ---------------------------------------------------------------------------
# Keyspace expiry listener — daemon thread
# ---------------------------------------------------------------------------

def keyspace_listener():
    """Subscribe to Redis key-expired events; delete image files on TTL expiry."""
    # Needs its own connection — pub/sub blocks the connection.
    ps_rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = ps_rdb.pubsub()
    pubsub.psubscribe("__keyevent@0__:expired")
    for message in pubsub.listen():
        if message["type"] not in ("pmessage", "message"):
            continue
        key = message.get("data", "")
        if isinstance(key, str) and key.startswith("flux:job:"):
            delete_images(key[len("flux:job:"):])


# ---------------------------------------------------------------------------
# Worker proxy
# ---------------------------------------------------------------------------

def post_to_worker(job_input: dict) -> dict:
    """POST to /runsync on the handler; retry until the worker is reachable."""
    while True:
        try:
            resp = requests.post(
                f"{WORKER_URL}/runsync",
                json={"input": job_input},
                timeout=WORKER_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            print(
                f"[redis_worker] Worker not reachable at {WORKER_URL}, "
                f"retrying in {WORKER_RETRY_DELAY}s…",
                flush=True,
            )
            time.sleep(WORKER_RETRY_DELAY)


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------

def process_job(job_id: str):
    job_key = f"flux:job:{job_id}"

    # Skip jobs cancelled while waiting in the queue.
    status = rdb.hget(job_key, "status")
    if status != "IN_QUEUE":
        print(f"[redis_worker] Skipping job {job_id} (status={status})", flush=True)
        return

    rdb.hset(job_key, mapping={"status": "IN_PROGRESS", "started_at": str(time.time())})
    print(f"[redis_worker] Processing job {job_id}", flush=True)

    try:
        job_input = json.loads(rdb.hget(job_key, "input"))
        result = post_to_worker(job_input)

        if result.get("status") == "FAILED":
            raise RuntimeError(result.get("output", {}).get("error", "Worker returned FAILED"))

        output = result.get("output", {})
        images = output.get("images", [])

        # Persist images and replace data-URIs with serve paths.
        save_images(job_id, images)
        output["images"]    = [f"/image/{job_id}/{i}" for i in range(len(images))]
        output["image_url"] = output["images"][0] if output["images"] else ""

        rdb.hset(job_key, mapping={
            "status":       "COMPLETED",
            "output":       json.dumps(output),
            "completed_at": str(time.time()),
        })
        rdb.expire(job_key, JOB_TTL)
        print(f"[redis_worker] Job {job_id} completed (TTL={JOB_TTL}s)", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        rdb.hset(job_key, mapping={
            "status":       "FAILED",
            "error":        str(exc),
            "completed_at": str(time.time()),
        })
        rdb.expire(job_key, JOB_TTL)
        print(f"[redis_worker] Job {job_id} FAILED: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"[redis_worker] Redis:      {REDIS_URL}", flush=True)
    print(f"[redis_worker] Worker URL: {WORKER_URL}", flush=True)

    # Start keyspace expiry listener in the background.
    threading.Thread(target=keyspace_listener, daemon=True).start()

    print("[redis_worker] Waiting for jobs…", flush=True)
    while True:
        result = rdb.blpop(QUEUE_KEY, timeout=0)
        if result is None:
            continue
        _, job_id = result
        process_job(job_id)


if __name__ == "__main__":
    main()


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

"""Test the local FLUX.1-schnell worker running via Docker Compose.

Usage:
    # Build and start all services (first run downloads ~24 GB of weights):
    #   docker compose up --build
    #
    # Then run (set API_KEY in .env or pass it directly):
    #   python test_local.py
    #   API_KEY=my-secret python test_local.py

The API gateway is at http://localhost:8000.
Jobs are submitted to POST /run, then polled via GET /status/{id}.
"""

import os
import base64
import time
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("LOCAL_SERVER_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

POLL_INTERVAL = 2    # seconds between status checks
POLL_TIMEOUT  = 300  # seconds before giving up


def generate(payload: dict) -> dict:
    """Submit a job, poll until done, and return the output dict."""
    resp = requests.post(f"{BASE_URL}/run", json={"input": payload}, headers=HEADERS)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"  Job submitted: {job_id}")

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data["status"]
        print(f"  Status: {status}")
        if status == "COMPLETED":
            return data.get("output") or {}
        if status == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error')}")
        if status == "CANCELLED":
            raise RuntimeError("Job was cancelled")

    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT}s")


def save_images(result: dict, prefix: str = "output"):
    """Download images from the API and save them to disk."""
    images = result.get("images", [])
    if not images:
        print("No images in result.")
        return

    os.makedirs("test_outputs", exist_ok=True)
    for i, img_data in enumerate(images):
        path = os.path.join("test_outputs", f"{prefix}_{i}.png")
        if isinstance(img_data, str) and img_data.startswith("/image/"):
            # Images are stored on the server — download via the API.
            resp = requests.get(f"{BASE_URL}{img_data}", headers=HEADERS)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"  Saved {path}")
        elif isinstance(img_data, str) and img_data.startswith("data:image"):
            b64 = img_data.split(",", 1)[1]
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"  Saved {path}")
        elif isinstance(img_data, str) and img_data.startswith("http"):
            print(f"  Image URL: {img_data}")
        else:
            print(f"  Unknown image format: {str(img_data)[:120]}")


def fetch_as_base64(img_data: str) -> str:
    """Return a data-URI base64 string from either an API path or an existing data-URI."""
    if img_data.startswith("/image/"):
        resp = requests.get(f"{BASE_URL}{img_data}", headers=HEADERS)
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return img_data  # already a data-URI


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_test_mask(width: int, height: int) -> str:
    from PIL import Image, ImageDraw

    mask = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(mask)
    x0, y0 = int(width * 0.3), int(height * 0.3)
    x1, y1 = int(width * 0.7), int(height * 0.7)
    draw.rectangle([x0, y0, x1, y1], fill="white")
    buf = BytesIO()
    mask.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_txt2img():
    print("\n=== Test: txt2img ===")
    result = generate({
        "prompt": "A futuristic city at sunset, cyberpunk style",
        "width": 720,
        "height": 1280,
        "seed": 42,
        "num_inference_steps": 4,
        "num_images": 1,
    })
    print(f"  Seed: {result.get('seed')}")
    save_images(result, prefix="txt2img")
    return result


def test_img2img(reference_image_path: str = None):
    print("\n=== Test: img2img ===")

    if reference_image_path and os.path.exists(reference_image_path):
        image_b64 = image_to_base64(reference_image_path)
    else:
        print("  No reference image found, generating one via txt2img first...")
        txt_result = test_txt2img()
        images = txt_result.get("images", [])
        if not images:
            print("  Failed to generate reference image.")
            return
        image_b64 = fetch_as_base64(images[0])

    result = generate({
        "mode": "img2img",
        "prompt": "Remove all background, leave only the central object. Then replace the background with a futuristic city at sunset, cyberpunk style",
        "image": image_b64,
        "strength": 0.75,
        "width": 1280,
        "height": 1280,
        "num_inference_steps": 4,
        "num_images": 1,
    })
    print(f"  Seed: {result.get('seed')}")
    save_images(result, prefix="img2img")
    return result


def test_inpainting(reference_image_path: str = None):
    print("\n=== Test: inpainting ===")

    if reference_image_path and os.path.exists(reference_image_path):
        image_b64 = image_to_base64(reference_image_path)
    else:
        print("  No reference image found, generating one via txt2img first...")
        txt_result = test_txt2img()
        images = txt_result.get("images", [])
        if not images:
            print("  Failed to generate reference image.")
            return
        image_b64 = fetch_as_base64(images[0])

    mask_b64 = create_test_mask(720, 1280)
    result = generate({
        "mode": "inpainting",
        "prompt": "A glowing magical portal with blue energy",
        "image": image_b64,
        "mask_image": mask_b64,
        "strength": 0.85,
        "width": 720,
        "height": 1280,
        "num_inference_steps": 4,
        "num_images": 1,
    })
    print(f"  Seed: {result.get('seed')}")
    save_images(result, prefix="inpaint")
    return result


if __name__ == "__main__":
    test_txt2img()

    ref_path = "test_img/tiger-tank.jpg"
    test_img2img(ref_path)
    # test_inpainting(ref_path)

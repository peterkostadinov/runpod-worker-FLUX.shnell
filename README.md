# RunPod Serverless Worker — FLUX.1-schnell

Run [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) as a RunPod serverless endpoint with **text-to-image**, **image-to-image**, and **inpainting** support.

Uses the official `black-forest-labs/FLUX.1-schnell` model via [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) — no third-party wrappers, no telemetry, minimal dependencies.

FLUX.1-schnell is a distilled variant of FLUX.1 that produces high-quality images in just **4 inference steps**, making it significantly faster than FLUX.1-dev. It uses no classifier-free guidance and does not require negative prompts. Licensed under **Apache 2.0**.

### Memory optimization

The transformer and T5 text encoder are quantized to **int8** via [optimum-quanto](https://github.com/huggingface/optimum-quanto), reducing VRAM from ~34 GB to ~17.5 GB. This allows the model to run on **24 GB GPUs** (RTX 3090, RTX 4090) with room for activations. All three pipelines (txt2img, img2img, inpainting) share the same weights — zero extra VRAM.

---

## Project Structure

```
handler.py           — RunPod serverless handler (model loading + inference)
schemas.py           — Input validation schema
download_weights.py  — Pre-download model weights (optional, for Docker build caching)
Dockerfile           — Container image definition
docker-compose.yml   — Run the worker locally with GPU passthrough
requirements.txt     — Python dependencies
.env.example         — Environment variable template
test_endpoint.py     — Test script for the cloud RunPod endpoint
test_local.py        — Test script for the local Docker worker
local_server.py      — Standalone FastAPI server (no Docker, requires local CUDA PyTorch)
```

---

## Modes

The worker supports three generation modes via the `mode` parameter:

| Mode          | Description                                           | Requires                    |
| :------------ | :---------------------------------------------------- | :-------------------------- |
| `txt2img`     | Generate an image from a text prompt (default)        | `prompt`                    |
| `img2img`     | Transform a reference image guided by a text prompt   | `prompt`, `image`           |
| `inpainting`  | Fill a masked region of an image guided by a prompt   | `prompt`, `image`, `mask_image` |

---

## Input Parameters

| Parameter             | Type    | Default    | Required | Description                                                              |
| :-------------------- | :------ | :--------- | :------- | :----------------------------------------------------------------------- |
| `mode`                | `str`   | `txt2img`  | No       | Generation mode: `txt2img`, `img2img`, or `inpainting`                   |
| `prompt`              | `str`   | `None`     | **Yes**  | Text prompt describing the desired image                                 |
| `image`               | `str`   | `None`     | img2img/inpainting | Base64-encoded reference image (data URI or raw base64)     |
| `mask_image`          | `str`   | `None`     | inpainting | Base64-encoded mask (white = area to inpaint, black = keep)            |
| `strength`            | `float` | `0.75`     | No       | How much to transform the reference image (0.0 = no change, 1.0 = full) |
| `height`              | `int`   | `1024`     | No       | Height of the generated image in pixels                                  |
| `width`               | `int`   | `1024`     | No       | Width of the generated image in pixels                                   |
| `seed`                | `int`   | `None`     | No       | Random seed for reproducibility. If `None`, a random seed is generated   |
| `num_inference_steps` | `int`   | `4`        | No       | Number of denoising steps (schnell is optimized for 4 steps)             |
| `guidance_scale`      | `float` | `0.0`      | No       | Classifier-free guidance scale (should be 0.0 for schnell)               |
| `num_images`          | `int`   | `1`        | No       | Number of images to generate per prompt (1 or 2)                         |

## Output Fields

| Field       | Type     | Description                                                                 |
| :---------- | :------- | :-------------------------------------------------------------------------- |
| `image_url` | `str`    | The first generated image (base64 data URI, or S3 URL if bucket configured) |
| `images`    | `str[]`  | Array of all generated images                                               |
| `seed`      | `int`    | The seed used (useful for reproducibility)                                  |

---

## Usage

### Text-to-Image (default)

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a tiny astronaut hatching from an egg on the moon",
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 4,
      "seed": 42
    }
  }'
```

### Image-to-Image

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "img2img",
      "prompt": "same scene in watercolor painting style",
      "image": "data:image/png;base64,iVBORw0KGgo...",
      "strength": 0.75,
      "width": 720,
      "height": 1280,
      "num_inference_steps": 4
    }
  }'
```

### Inpainting

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "inpainting",
      "prompt": "a glowing magical portal with blue energy",
      "image": "data:image/png;base64,iVBORw0KGgo...",
      "mask_image": "data:image/png;base64,iVBORw0KGgo...",
      "strength": 0.85,
      "width": 720,
      "height": 1280,
      "num_inference_steps": 4
    }
  }'
```

> **Mask format:** White pixels = area to regenerate, black pixels = area to preserve.

### Check Job Status

```bash
curl "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/{JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

| Status        | Meaning                                       |
| :------------ | :-------------------------------------------- |
| `IN_QUEUE`    | Job is waiting for a worker                    |
| `IN_PROGRESS` | Worker is generating                           |
| `COMPLETED`   | Done — output is available                     |
| `FAILED`      | An error occurred (check `error` field)        |

### Python Example

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Text-to-image
result = endpoint.run_sync({
    "prompt": "a tiny astronaut hatching from an egg on the moon",
    "width": 720,
    "height": 1280,
    "num_inference_steps": 4,
    "seed": 42,
})

# Save the image
img_data = result["image_url"].split(",")[1]
with open("output.png", "wb") as f:
    f.write(base64.b64decode(img_data))

# Image-to-image (reuse the generated image)
with open("output.png", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

result = endpoint.run_sync({
    "mode": "img2img",
    "prompt": "same scene in oil painting style",
    "image": f"data:image/png;base64,{ref_b64}",
    "strength": 0.75,
})
```

### JavaScript Example

```javascript
const response = await fetch(
  `https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`,
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: {
        prompt: "a tiny astronaut hatching from an egg on the moon",
        width: 720,
        height: 1280,
        num_inference_steps: 4,
        seed: 42,
      },
    }),
  }
);

const { output } = await response.json();
console.log("Image URL:", output.image_url);
```

---

## Local Development (Docker)

The easiest way to run the worker locally on a GPU machine is via Docker Compose — the image has CUDA PyTorch baked in, so there are no local Python environment requirements.

### Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- An NVIDIA GPU with 24 GB+ VRAM (RTX 3090 / 4090 recommended)
- A Hugging Face account with access to [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

### Setup

```bash
# 1. Copy the env template and fill in your tokens
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_...   ← required — FLUX.1-schnell is a gated model
#   API_KEY=...       ← optional, only needed for local_server.py

# 2. Build the image and start the worker
#    (first run downloads ~24 GB of model weights — cached in a Docker volume)
docker compose up --build
```

The RunPod SDK detects it is running outside of RunPod infrastructure and starts a local HTTP server on **port 8000**. Model weights are persisted in a named Docker volume (`hf_cache`) so they are only downloaded once.

### Test the local worker

```bash
python test_local.py
```

This hits `POST /runsync` with `{"input": {...}}` — exactly the same handler function and payload format used in production. Results are saved to `test_outputs/`.

### Local server HTTP API

| Method | Path       | Auth         | Description                        |
| :----- | :--------- | :----------- | :--------------------------------- |
| `GET`  | `/health`  | None         | Liveness check                     |
| `POST` | `/runsync` | None         | Synchronous generation (RunPod SDK)|

**Example:**

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a tiny astronaut hatching from an egg on the moon",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 4,
      "seed": 42
    }
  }'
```

---

## Deployment

1. Copy and configure environment variables:
   ```bash
   cp .env.example .env
   ```
2. Build the Docker image:
   ```bash
   docker build -t flux-schnell-worker .
   ```
3. Push to your container registry.
4. Deploy as a RunPod serverless endpoint.

### Environment Variables

| Variable              | Default                              | Description                                                      |
| :-------------------- | :----------------------------------- | :--------------------------------------------------------------- |
| `HF_TOKEN`            | *(unset)*                            | Hugging Face access token (**required** — FLUX.1-schnell is gated) |
| `HF_MODEL`            | `black-forest-labs/FLUX.1-schnell`   | Hugging Face model ID to load                                    |
| `API_KEY`             | *(auto-generated)*                   | Bearer token for `local_server.py` (not needed for Docker worker)|
| `BUCKET_ENDPOINT_URL` | *(unset)*                            | S3-compatible bucket URL for image uploads instead of base64     |

### Pipeline Settings

| Setting               | Value            | Reason                                                  |
| :-------------------- | :--------------- | :------------------------------------------------------ |
| `torch_dtype`         | `bfloat16`       | Native dtype for FLUX; halves VRAM vs float32           |
| `guidance_scale`      | `0.0` (default)  | Schnell is distilled — CFG is not used                  |
| `max_sequence_length` | `256`            | Saves ~1-2 GB VRAM; schnell prompts rarely exceed this  |
| Quantization          | `int8` (quanto)  | Halves transformer + T5 VRAM; fits in 24 GB GPUs        |

---

## Dependencies

- **PyTorch 2.7** + CUDA 12.1
- **Diffusers** — Hugging Face diffusion pipelines (txt2img, img2img, inpainting)
- **Transformers** — tokenizer / text encoder
- **Accelerate** — efficient model loading
- **optimum-quanto** — int8 weight quantization
- **xformers** — memory-efficient attention
- **RunPod SDK** — serverless handler

See [requirements.txt](requirements.txt) for the full list.

---

## License

This project is licensed under the [MIT License](LICENSE). The FLUX.1-schnell model itself is licensed under [Apache 2.0](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md).

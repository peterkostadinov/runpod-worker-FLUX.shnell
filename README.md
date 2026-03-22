# RunPod Serverless Worker — FLUX.1-schnell

Run [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) as a RunPod serverless endpoint to generate images.

Uses the official `black-forest-labs/FLUX.1-schnell` model via [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) — no third-party wrappers, no telemetry, minimal dependencies.

FLUX.1-schnell is a distilled variant of FLUX.1 that produces high-quality images in just **4 inference steps**, making it significantly faster than FLUX.1-dev. It uses no classifier-free guidance and does not require negative prompts. Licensed under **Apache 2.0**.

---

## Project Structure

```
handler.py           — RunPod serverless handler (model loading + inference)
schemas.py           — Input validation schema
download_weights.py  — Pre-download model weights (for Docker build caching)
Dockerfile           — Container image definition
requirements.txt     — Python dependencies
```

---

## Input Parameters

| Parameter             | Type  | Default | Required | Description                                                          |
| :-------------------- | :---- | :------ | :------- | :------------------------------------------------------------------- |
| `prompt`              | `str` | `None`  | **Yes**  | The main text prompt describing the desired image.                   |
| `height`              | `int` | `1024`  | No       | The height of the generated image in pixels                          |
| `width`               | `int` | `1024`  | No       | The width of the generated image in pixels                           |
| `seed`                | `int` | `None`  | No       | Random seed for reproducibility. If `None`, a random seed is generated |
| `num_inference_steps` | `int` | `4`     | No       | Number of denoising steps (schnell is optimized for 4 steps)         |
| `num_images`          | `int` | `1`     | No       | Number of images to generate per prompt (must be 1 or 2)             |

## Output Fields

| Field       | Type     | Description                                                                 |
| :---------- | :------- | :-------------------------------------------------------------------------- |
| `image_url` | `str`    | The first generated image (base64 data URI, or S3 URL if bucket configured) |
| `images`    | `str[]`  | Array of all generated images                                               |
| `seed`      | `int`    | The seed used (useful for reproducibility)                                  |

---

## Usage

### Submit a Request (`/run`)

Send a POST request to start an image generation job. This is **asynchronous** — it returns a job `id` immediately.

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
      "seed": 42,
      "num_images": 1
    }
  }'
```

**Response:**

```json
{
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "status": "IN_QUEUE"
}
```

> **Tip:** For quick, blocking requests you can use `/runsync` instead of `/run`. It waits for the job to finish and returns the output directly — but it will time out after **30 seconds**.

### Check Job Status (`/status`)

Poll the status endpoint with the job `id` to track progress:

```bash
curl "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/{JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

**Possible statuses:**

| Status        | Meaning                                       |
| :------------ | :-------------------------------------------- |
| `IN_QUEUE`    | Job is waiting for a worker to become available |
| `IN_PROGRESS` | A worker is currently generating images        |
| `COMPLETED`   | Done — output is available                     |
| `FAILED`      | An error occurred (check `error` field)        |

### Get the Result

Once the status is `COMPLETED`, the response includes the output:

```json
{
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "status": "COMPLETED",
  "delayTime": 2500,
  "executionTime": 1200,
  "workerId": "462u6mrq9s28h6",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUh...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
    ],
    "seed": 42
  }
}
```

**To save the image** (base64 response):

```bash
echo "<base64_string>" | base64 -d > output.png
```

### Cancel a Job (optional)

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/cancel/{JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

### Python Example

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

run = endpoint.run_sync({
    "prompt": "a tiny astronaut hatching from an egg on the moon",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "seed": 42,
    "num_images": 1
})

# Save the image
img_data = run["image_url"].split(",")[1]
with open("output.png", "wb") as f:
    f.write(base64.b64decode(img_data))
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
        height: 1024,
        width: 1024,
        num_inference_steps: 4,
        seed: 42,
        num_images: 1,
      },
    }),
  }
);

const { output } = await response.json();
console.log("Image URL:", output.image_url);
```

---

## Deployment

1. Build the Docker image:
   ```bash
   docker build -t flux-schnell-worker .
   ```
2. Push to your container registry.
3. Deploy as a RunPod serverless endpoint.

### Environment Variables

| Variable              | Default                              | Description                                                      |
| :-------------------- | :----------------------------------- | :--------------------------------------------------------------- |
| `HF_MODEL`            | `black-forest-labs/FLUX.1-schnell`   | Hugging Face model ID to load                                    |
| `BUCKET_ENDPOINT_URL` | *(unset)*                            | S3-compatible bucket URL for image uploads instead of base64     |
| `HF_TOKEN`            | *(unset)*                            | Hugging Face access token (only needed for gated/private models) |

### Pipeline Settings (hardcoded)

| Setting          | Value          | Reason                                              |
| :--------------- | :------------- | :-------------------------------------------------- |
| `torch_dtype`    | `bfloat16`     | Native dtype for FLUX; halves VRAM vs float32       |
| `guidance_scale` | `0.0`          | Schnell is distilled — CFG is not used              |

---

## Dependencies

Minimal dependency footprint — only what's needed:

- **PyTorch 2.7** + CUDA 12.1
- **Diffusers** — Hugging Face diffusion pipeline
- **Transformers** — tokenizer / text encoder
- **Accelerate** — efficient model loading
- **xformers** — memory-efficient attention
- **RunPod SDK** — serverless handler

See [requirements.txt](requirements.txt) for the full list.

---

## License

This project is licensed under the [MIT License](LICENSE). The FLUX.1-schnell model itself is licensed under [Apache 2.0](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md).

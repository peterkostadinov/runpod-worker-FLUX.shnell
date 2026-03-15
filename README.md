# RunPod Serverless Worker — FLUX.1-schnell

> **Based on**: [PrunaAI/runpod-worker-FLUX.1-dev](https://github.com/PrunaAI/runpod-worker-FLUX.1-dev) by [PrunaAI](https://pruna.ai). Original optimization and worker architecture by PrunaAI.

---

Run an optimized [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) as a RunPod serverless endpoint to generate images.

FLUX.1-schnell is a distilled variant of FLUX.1 that produces high-quality images in just **4 inference steps**, making it significantly faster than FLUX.1-dev. It uses no classifier-free guidance and does not require negative prompts.

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type  | Default | Required | Description                                                          |
| :-------------------- | :---- | :------ | :------- | :------------------------------------------------------------------- |
| `prompt`              | `str` | `None`  | **Yes**  | The main text prompt describing the desired image.                   |
| `height`              | `int` | `1024`  | No       | The height of the generated image in pixels                          |
| `width`               | `int` | `1024`  | No       | The width of the generated image in pixels                           |
| `seed`                | `int` | `None`  | No       | Random seed for reproducibility. If `None`, a random seed is generated |
| `num_inference_steps` | `int` | `4`     | No       | Number of denoising steps (schnell is optimized for 4 steps)         |
| `num_images`          | `int` | `1`     | No       | Number of images to generate per prompt (Constraint: must be 1 or 2) |

### Step 1 — Submit a Request (`/run`)

Send a POST request to start an image generation job. This is **asynchronous** — it returns a job `id` immediately.

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a knitted purple prune",
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

### Step 2 — Check Job Status (`/status`)

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

### Step 3 — Get the Result

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

- **`image_url`** — The first generated image (base64 data URI or S3 URL if bucket storage is configured).
- **`images`** — Array of all generated images (when `num_images` > 1).
- **`seed`** — The seed used (useful for reproducibility).

**To save the image** (base64 response):

```bash
# Extract the base64 data and decode it to a file
echo "<base64_string>" | base64 -d > output.png
```

### Step 4 — Cancel a Job (optional)

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

# Submit and wait for the result
run = endpoint.run_sync({
    "prompt": "a knitted purple prune",
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
        prompt: "a knitted purple prune",
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

The default model is `PrunaAI/FLUX.1-schnell-smashed-no-compile`. Override via the `HF_MODEL` environment variable.

### Environment Variables

| Variable              | Description                                                    |
| :-------------------- | :------------------------------------------------------------- |
| `HF_MODEL`            | Hugging Face model ID (default: `PrunaAI/FLUX.1-schnell-smashed-no-compile`) |
| `BUCKET_ENDPOINT_URL` | S3-compatible bucket URL for uploading images instead of returning base64 |

---

## Credits

This project is based on the [FLUX.1-dev RunPod worker](https://github.com/PrunaAI/runpod-worker-FLUX.1-dev) by [PrunaAI](https://pruna.ai), adapted for FLUX.1-schnell.

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

### Example Request

```json
{
  "input": {
    "prompt": "a knitted purple prune",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "seed": 42,
    "num_images": 1
  }
}
```

which produces output like this:

```json
{
  "delayTime": 2500,
  "executionTime": 1200,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU..."
    ],
    "seed": 42
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
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

---

## Credits

This project is based on the [FLUX.1-dev RunPod worker](https://github.com/PrunaAI/runpod-worker-FLUX.1-dev) by [PrunaAI](https://pruna.ai), adapted for FLUX.1-schnell.

# FLUX.1-dev → FLUX.1-schnell Conversion Plan

## Overview

Convert this RunPod serverless worker from **FLUX.1-dev** to **FLUX.1-schnell**. FLUX.1-schnell is a distilled, faster variant that requires fewer inference steps and **no classifier-free guidance** (guidance_scale ≈ 0). It also does **not use negative prompts**.

> **Original project**: Based on [PrunaAI/runpod-worker-FLUX.1-dev](https://github.com/PrunaAI/runpod-worker-FLUX.1-dev) by [PrunaAI](https://pruna.ai).

---

## Key Differences: FLUX.1-dev vs FLUX.1-schnell

| Aspect | FLUX.1-dev | FLUX.1-schnell |
|---|---|---|
| HuggingFace model | `PrunaAI/FLUX.1-dev-smashed-no-compile` | `PrunaAI/FLUX.1-schnell-smashed-no-compile` |
| Default inference steps | 25 | 4 |
| Guidance scale | 7.5 | 0.0 (no CFG) |
| Negative prompt | Supported | Not used (ignored when guidance_scale=0) |
| Typical generation speed | ~6s (compiled) | ~1-2s (compiled) |
| License | Non-commercial | Apache 2.0 |

---

## Step-by-Step Implementation Plan

### Step 1: Update `download_weights.py`

**File**: `download_weights.py`

**Changes**:
1. Change the default `HF_MODEL` env var from `"PrunaAI/FLUX.1-dev-smashed-no-compile"` to `"PrunaAI/FLUX.1-schnell-smashed-no-compile"`.
2. Update the docstring from "FLUX.1-dev" to "FLUX.1-schnell".

**Before**:
```python
def get_diffusion_pipelines():
    """
    Fetches the FLUX.1-dev pipeline from the HuggingFace model hub.
    """
    pipe = fetch_pretrained_model(
        os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-dev-smashed-no-compile")
    )
```

**After**:
```python
def get_diffusion_pipelines():
    """
    Fetches the FLUX.1-schnell pipeline from the HuggingFace model hub.
    """
    pipe = fetch_pretrained_model(
        os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-schnell-smashed-no-compile")
    )
```

---

### Step 2: Update `handler.py`

**File**: `handler.py`

There are **4 separate edits** in this file. Apply each one individually.

#### Edit 2a: Update `load_models` comment and model ID (~line 21-24)

**Before**:
```python
    def load_models(self):
        # Load FLUX.1-dev pipeline from cache using identifier

        self.pipe = PrunaModel.from_hub(
            os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-dev-smashed-no-compile"),
            local_files_only=True,
        )
```

**After**:
```python
    def load_models(self):
        # Load FLUX.1-schnell pipeline from cache using identifier

        self.pipe = PrunaModel.from_hub(
            os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-schnell-smashed-no-compile"),
            local_files_only=True,
        )
```

#### Edit 2b: Update `generate_image` docstring (~line 56)

**Before**:
```python
def generate_image(job):
    """
    Generate an image from text using FLUX.1-dev Model
    """
```

**After**:
```python
def generate_image(job):
    """
    Generate an image from text using FLUX.1-schnell Model
    """
```

#### Edit 2c: Update pipeline call comment (~line 107)

**Before**:
```python
        # Generate image using FLUX.1-dev pipeline
```

**After**:
```python
        # Generate image using FLUX.1-schnell pipeline
```

#### Edit 2d: Update pipeline call — remove `negative_prompt`, hardcode `guidance_scale=0.0` (~lines 108-116)

**Before**:
```python
            result = MODELS.pipe(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
```

**After**:
```python
            result = MODELS.pipe(
                prompt=job_input["prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=0.0,
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
```

---

### Step 3: Update `schemas.py`

**File**: `schemas.py`

**Changes**:
1. Remove the `negative_prompt` field (not used by schnell).
2. Change `num_inference_steps` default from `25` to `4`.
3. Remove the `guidance_scale` field (always 0.0 for schnell).

**Full replacement**:
```python
INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 4
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
}
```

---

### Step 4: Update `test_input.json`

**File**: `test_input.json`

**Changes**:
1. Remove `negative_prompt` field.
2. Change `num_inference_steps` from `25` to `4`.
3. Remove `guidance_scale` field.

**Full replacement**:
```json
{
  "input": {
    "prompt": "A majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "seed": 42,
    "num_images": 1
  }
}
```

---

### Step 5: Update `.runpod/hub.json`

**File**: `.runpod/hub.json`

**Changes**:
1. Change `title` from `"FLUX.1-dev-juiced"` to `"FLUX.1-schnell-juiced"`.
2. Change `description` to reference FLUX.1-schnell.
3. Update presets to show only schnell presets (or make schnell the default).
4. Change the default `HF_MODEL` env value from `"PrunaAI/FLUX.1-dev-smashed-no-compile"` to `"PrunaAI/FLUX.1-schnell-smashed-no-compile"`.

**Full replacement**:
```json
{
  "title": "FLUX.1-schnell-juiced",
  "description": "Generate images with a Pruna AI optimized FLUX.1-schnell.",
  "type": "serverless",
  "category": "image",
  "iconUrl": "https://framerusercontent.com/images/1zpNZDseKMJxNFweAzbVFUyrn8.svg",
  "config": {
    "runsOn": "GPU",
    "containerDiskInGb": 80,
    "gpuIds": "AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO",
    "gpuCount": 1,
    "allowedCudaVersions": [
      "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1"
    ],
    "presets": [
      {
        "name": "FLUX.1-schnell-juiced (no compilation)",
        "defaults": {
          "HF_MODEL": "PrunaAI/FLUX.1-schnell-smashed-no-compile"
        }
      },
      {
        "name": "FLUX.1-schnell-juiced (compilation)",
        "defaults": {
          "HF_MODEL": "PrunaAI/FLUX.1-schnell-smashed"
        }
      }
    ],
    "env": [
      {
        "key": "HF_MODEL",
        "input": {
          "type": "huggingface",
          "name": "Hugging Face Model",
          "description": "Pruna AI FLUX model and organization/name as listed on Huggingface Hub",
          "default": "PrunaAI/FLUX.1-schnell-smashed-no-compile"
        }
      }
    ]
  }
}
```

---

### Step 6: Update `Dockerfile`

**File**: `Dockerfile`

**No code changes required**. The Dockerfile is model-agnostic — it installs dependencies, copies files, and runs `download_weights.py` which reads the `HF_MODEL` env var. The schnell model weights will be downloaded automatically via the updated `download_weights.py`.

---

### Step 7: Update `requirements.txt`

**File**: `requirements.txt`

**No changes required**. The same dependencies (pruna, torch, runpod, etc.) are used for both FLUX.1-dev and FLUX.1-schnell.

---

### Step 8: Update `README.md`

**File**: `README.md`

> **ALREADY DONE — NO ACTION NEEDED.** The README has already been rewritten for FLUX.1-schnell with full attribution to the original PrunaAI project. Verify it looks correct but do not re-edit.

---

### Step 9: Update GitHub CI/CD Workflows (Optional)

**Files**: `.github/workflows/*.yml`

**No changes required**. The workflows are generic (build Docker, push, test) and do not reference FLUX.1-dev or schnell specifically. They use repository variables (`${{ vars.DOCKERHUB_REPO }}`, etc.) which should be updated in the GitHub repository settings if the Docker image name changes.

> **Note**: If deploying under a different Docker Hub image name (e.g., `worker-flux1-schnell` instead of `worker-flux1-dev`), update the `DOCKERHUB_IMG` repository variable in GitHub Settings → Secrets and variables → Actions.

---

## Files Changed Summary

| File | Action | Description |
|---|---|---|
| `download_weights.py` | **Edit** | Change model ID to schnell, update docstring |
| `handler.py` | **Edit** | Change model ID, remove negative_prompt/guidance_scale from pipeline call, update comments |
| `schemas.py` | **Edit** | Remove negative_prompt & guidance_scale fields, change steps default to 4 |
| `test_input.json` | **Edit** | Remove negative_prompt & guidance_scale, change steps to 4 |
| `.runpod/hub.json` | **Edit** | Change title, description, presets, and default model to schnell |
| `README.md` | **Already done** | Rewritten for schnell with PrunaAI attribution |
| `Dockerfile` | No change | Model-agnostic |
| `requirements.txt` | No change | Same dependencies |
| `.github/workflows/*` | No change | Generic CI/CD, uses repo variables |

---

## Verification Checklist

After implementing all changes, verify:

- [ ] `grep -r "FLUX.1-dev" .` returns no results (except in planning docs and attribution)
- [ ] `grep -r "FLUX.1-schnell" .` shows the new model references
- [ ] `python3 -c "from schemas import INPUT_SCHEMA; print(INPUT_SCHEMA)"` shows no negative_prompt or guidance_scale
- [ ] `cat test_input.json | python3 -m json.tool` validates JSON structure
- [ ] `cat .runpod/hub.json | python3 -m json.tool` validates JSON structure
- [ ] Docker build succeeds: `docker build -t flux-schnell-worker .`
- [ ] Test with: `python3 handler.py --test_input test_input.json`

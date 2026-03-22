import os

import torch
from diffusers import FluxPipeline


def fetch_pretrained_model(model_name):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return FluxPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def get_diffusion_pipelines():
    """
    Fetches the FLUX.1-schnell pipeline from the HuggingFace model hub.
    """
    pipe = fetch_pretrained_model(
        os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
    )

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()

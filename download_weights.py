import os

from pruna import PrunaModel


def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return PrunaModel.from_hub(model_name, **kwargs)
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
        os.environ.get("HF_MODEL", "PrunaAI/FLUX.1-schnell-smashed-no-compile")
    )

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()

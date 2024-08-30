import huggingface_hub
import os

MIMI_FILENAME = "mimi_0abbed5f@100.safetensors"
TOKENIZER_FILENAME = "tokenizer-de0e421d-checkpoint40.safetensors"
TOKENIZER_SPM_FILENAME = "tokenizer_spm_32k_3.model"


def download_moshi_models(*, path: str, force: bool):
    abs_path = os.path.abspath(path)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        print("Please set the HF_TOKEN environment variable.")
        exit(1)

    huggingface_hub.hf_hub_download(
        "klk42/msh-v0.1",
        MIMI_FILENAME,
        token=hf_token,
        local_dir=abs_path,
        force_download=force,
    )

    huggingface_hub.hf_hub_download(
        "klk42/msh-v0.1",
        TOKENIZER_FILENAME,
        token=hf_token,
        local_dir=abs_path,
        force_download=force,
    )

    huggingface_hub.hf_hub_download(
        "klk42/msh-v0.1",
        TOKENIZER_SPM_FILENAME,
        token=hf_token,
        local_dir=abs_path,
        force_download=force,
    )

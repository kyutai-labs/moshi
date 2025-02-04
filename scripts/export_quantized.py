import argparse
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from moshi.models import loaders
from moshi.modules.transformer import quantize_transformer

def main():
    parser = argparse.ArgumentParser('export_quantized')
    parser.add_argument('moshi_repo')

    args = parser.parse_args()

    token = open('token.txt').read().strip()
    api = HfApi(token=token)
    repo = args.moshi_repo
    new_repo = repo.rsplit('-', 1)[0] + '-q8-tmp'
    if not api.repo_exists(new_repo):
        api.create_repo(new_repo, repo_type='model')
        print("Repo created.")

    to_copy = ['README.md', 'tokenizer-e351c8d8-checkpoint125.safetensors', 'tokenizer_spm_32k_3.model']
    for file in to_copy:
        if not api.file_exists(new_repo, file):
            print("File", file, "is missing")
            old_file = hf_hub_download(repo, file)
            api.upload_file(
                path_or_fileobj=old_file,
                path_in_repo=file,
                repo_id=new_repo,
                repo_type="model")
    print("Downloading base model.")
    ckpt = hf_hub_download(repo, loaders.MOSHI_NAME)
    print("Creating model.")
    model = loaders.get_moshi_lm(ckpt, device='cuda')
    print("Quantizing model.")
    quantize_transformer(model)
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=True) as file:
        save_file(model.state_dict(), file.name)
        size = Path(file.name).stat().st_size / 1e9
        print(f"Checkpoint size: {size:.1f}GB")
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo='model.q8.safetensors',
            repo_id=new_repo,
            repo_type="model")


if __name__ == "__main__":
    main()

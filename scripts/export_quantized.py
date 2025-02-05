# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Convert a repo into a quantized one (PyTorch only). Need to run from a GPU."""


import argparse
import json
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from moshi.models import loaders
from moshi.modules.transformer import quantize_transformer


def get_api():
    token = input("Write token? ").strip()
    api = HfApi(token=token)
    return api


def main():
    parser = argparse.ArgumentParser('export_quantized')
    parser.add_argument('hf_repo')

    args = parser.parse_args()
    api = get_api()

    repo = args.hf_repo

    print("Downloading base model.")
    info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    print("Creating model.")
    model = info.get_moshi(device='cuda')
    print("Quantizing model.")
    quantize_transformer(model)

    new_repo = repo.rsplit('-', 1)[0] + '-q8'
    if not api.repo_exists(new_repo):
        api.create_repo(new_repo, repo_type='model')
        print("Repo created.")

    to_copy = ['README.md']
    for file in to_copy:
        if not api.file_exists(repo, file):
            continue
        if not api.file_exists(new_repo, file):
            print("File", file, "is missing")
            old_file = hf_hub_download(repo, file)
            api.upload_file(
                path_or_fileobj=old_file,
                path_in_repo=file,
                repo_id=new_repo,
                repo_type="model")
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=True) as file:
        save_file(model.state_dict(), file.name)
        size = Path(file.name).stat().st_size / 1e9
        print(f"Checkpoint size: {size:.1f}GB")
        old_name, old_ext = info.moshi_weights.name.rsplit('.', 1)
        new_name = old_name + '.q8.' + old_ext
        if False:
            api.upload_file(
                path_or_fileobj=file.name,
                path_in_repo=new_name,
                repo_id=new_repo,
                repo_type="model")
    config = json.load(open(hf_hub_download(repo, 'config.json')))
    config['moshi_name'] = new_name
    config['quantize'] = True
    if not config['mimi_name'].startswith('hf://'):
        config['mimi_name'] = f'hf://{repo}/{config["mimi_name"]}'
    if not config['tokenizer_name'].startswith('hf://'):
        config['tokenizer_name'] = f'hf://{repo}/{config["tokenizer_name"]}'
    with tempfile.NamedTemporaryFile(mode='w') as file:
        json.dump(config, file, indent=2)
        file.flush()
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo='config.json',
            repo_id=new_repo,
            repo_type="model")


if __name__ == "__main__":
    main()

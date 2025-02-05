# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Create or update a HF repo."""


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
    parser = argparse.ArgumentParser('update_repo')
    parser.add_argument('repo')
    parser.add_argument('-c', '--config')
    parser.add_argument('-r', '--readme')
    parser.add_argument('-m', '--mimi-weight')
    parser.add_argument('-M', '--moshi-weight')
    parser.add_argument('-t', '--tokenizer')

    args = parser.parse_args()

    api = get_api()
    if not api.repo_exists(args.repo):
        api.create_repo(args.repo, repo_type='model')
        print(f"Repo {args.repo} created.")

    old_config = None
    if api.file_exists(args.repo, 'config.json'):
        old_config = json.load(open(hf_hub_download(args.repo, 'config.json')))

    changes = False
    if args.config:
        changes = True
        new_config = json.load(open(args.config))
    elif old_config:
        new_config = old_config
    else:
        new_config = {}

    names = ['mimi_name', 'moshi_name', 'tokenizer_name']
    paths = [args.mimi_weight, args.moshi_weight, args.tokenizer]
    for name, path in zip(names, paths):
        if path is None:
            if old_config is not None and name in old_config:
                new_config[name] = old_config[name]
            continue
        filename = Path(path).name
        print(f"Uploading {path}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="model")
        new_config[name] = filename
        changes = True

    if changes:
        with tempfile.NamedTemporaryFile(mode='w') as file:
            json.dump(new_config, file, indent=2)
            file.flush()
            api.upload_file(
                path_or_fileobj=file.name,
                path_in_repo='config.json',
                repo_id=args.repo,
                repo_type="model")


if __name__ == "__main__":
    main()

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Export a model to HF."""


import argparse
import json
import tempfile

from huggingface_hub import HfApi

from moshi.models import loaders


def get_api():
    token = input("Write token? ").strip()
    api = HfApi(token=token)
    return api


def main():
    parser = argparse.ArgumentParser('export_quantized')
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")
    parser.add_argument('new_hf_repo')

    args = parser.parse_args()
    api = get_api()

    info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, moshi_weights=args.moshi_weight, mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer, config_path=args.config)

    if not api.repo_exists(args.new_hf_repo):
        api.create_repo(args.new_hf_repo, repo_type='model', private=True)
        print("Repo created.")

    config = info.raw_config
    assert config is not None
    config['mimi_name'] = info.mimi_weights.name
    config['moshi_name'] = info.moshi_weights.name
    config['tokenizer_name'] = info.tokenizer.name
    for file in [info.mimi_weights, info.moshi_weights, info.tokenizer]:
        if not api.file_exists(args.new_hf_repo, file.name):
            print("Uploading file", file)
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file.name,
                repo_id=args.new_hf_repo,
                repo_type="model")
    with tempfile.NamedTemporaryFile(mode='w') as file:
        json.dump(config, file, indent=2)
        file.flush()
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo='config.json',
            repo_id=args.new_hf_repo,
            repo_type="model")


if __name__ == "__main__":
    main()

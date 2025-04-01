# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from moshi.models import loaders
from pathlib import Path
import torch
from safetensors.torch import save_file

def import_model(
    tch_model,
    out_path: Path,
    silent: bool = False,
    max_out_n_q: int | None = None,
) -> None:
    in_n_q: int | None = None
    for idx in range(999):
        name = f"emb.{idx}.weight"
        if name not in tch_model:
            in_n_q = idx
            break
    out_n_q: int | None = None
    for idx in range(999):
        name = f"linears.{idx}.weight"
        if name not in tch_model:
            out_n_q = idx
            break
    assert in_n_q is not None
    assert out_n_q is not None
    if not silent:
        print(f"in_n_q: {in_n_q}, out_n_q: {out_n_q}")

    depformer_layers: int | None = None
    for idx in range(999):
        if f"depformer.layers.{idx}.self_attn.in_projs.0.weight" not in tch_model:
            depformer_layers = idx
            break
    assert depformer_layers is not None
    if not silent:
        print(f"depformer layers: {depformer_layers}")

    model = {}
    for name in ["text_emb.weight", "text_linear.weight"]:
        model[name] = tch_model[name]
    for name in tch_model.keys():
        if name.startswith("condition_provider.conditioners"):
            model[name] = tch_model[name]
    model["out_norm.weight"] = tch_model["out_norm.alpha"][0, 0]
    for idx in range(in_n_q):
        src_name = f"emb.{idx}.weight"
        dst_name = f"audio_embs.{idx}.weight"
        model[dst_name] = tch_model[src_name]

    for k, v in sorted(tch_model.items()):
        print(k, v.shape, v.dtype)
        if k.startswith("transformer"):
            if k.endswith(".alpha"):
                v = v[0, 0]
            k = k.replace(".alpha", ".weight")
            k = k.replace(".in_projs.0.weight", ".in_proj.weight")
            k = k.replace(".out_projs.0.weight", ".out_proj.weight")
            model[k] = v

    # Only export the first slices of the depformer (main).
    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        print(f"only exporting the first {exported_out_n_q} depformer layers")
    else:
        exported_out_n_q = out_n_q

    for idx in range(exported_out_n_q):
        tch_idx = idx

        base = f"depformer.slices.{idx}."
        model[base + "linear_in.weight"] = tch_model[f"depformer_in.{tch_idx}.weight"].clone()
        model[base + "linear_out.weight"] = tch_model[f"linears.{idx}.weight"]
        if idx == 0:
            model[base + "emb.weight"] = tch_model["depformer_text_emb.weight"]
            if "depformer_text_emb.low_rank.weight" in tch_model:
                model[base + "emb.low_rank.weight"] = tch_model["depformer_text_emb.low_rank.weight"].clone()
        else:
            model[base + "emb.weight"] = tch_model[f"depformer_emb.{idx-1}.weight"].clone()
            if f"depformer_emb.{tch_idx-1}.low_rank.weight" in tch_model:
                model[base + "emb.low_rank.weight"] = tch_model[f"depformer_emb.{idx-1}.low_rank.weight"].clone()

        for layer_idx in range(depformer_layers):
            layer = base + f"transformer.layers.{layer_idx}."
            model[layer + "self_attn.in_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.in_projs.{tch_idx}.weight"]
            )
            model[layer + "self_attn.out_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.out_projs.{tch_idx}.weight"]
            )
            model[layer + "norm1.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.norm1.alpha"
            ][0, 0].clone()
            model[layer + "norm2.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.norm2.alpha"
            ][0, 0].clone()
            model[layer + "gating.linear_in.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_in.weight"
            ].clone()
            model[layer + "gating.linear_out.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_out.weight"
            ].clone()

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument(
        "-s", "--silent", action="store_true", help="only prints the checkpoint name"
    )
    parser.add_argument(
        "--max-out-n-q",
        type=int,
        help="limit the number of depformer layers that are exported",
    )
    parser.add_argument("out", type=str, help="the mlx safetensors file to generate")

    args = parser.parse_args()
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path) 
    lm = checkpoint_info.get_moshi(device="cpu", dtype=torch.bfloat16, fuse_lora=True)
    for key, value in lm.state_dict().items():
        print(key, value.shape)
    out_path = Path(args.out)
    import_model(lm.state_dict(), out_path, silent=args.silent, max_out_n_q=args.max_out_n_q)

if __name__ == "__main__":
    main()

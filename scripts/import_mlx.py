# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file


def import_model(in_path: Path, out_path: Path, silent: bool = False) -> None:
    pkg = torch.load(in_path, map_location=torch.device("cpu"))
    tch_model = pkg["fsdp_best_state"]["model"]

    n_q: int | None = None
    for idx in range(999):
        name = f"emb.{idx}.weight"
        if name not in tch_model:
            n_q = idx
            break
    assert n_q is not None
    if not silent:
        print("Found n_q:", n_q)

    model = {}
    for name in ["text_emb.weight", "text_linear.weight"]:
        model[name] = tch_model[name]
    model["out_norm.weight"] = tch_model["out_norm.alpha"][0, 0]
    for idx in range(n_q):
        src_name = f"emb.{idx}.weight"
        dst_name = f"audio_embs.{idx}.weight"
        model[dst_name] = tch_model[src_name]

    for k, v in sorted(tch_model.items()):
        if not silent:
            print(k, v.shape, v.dtype)
        if k.startswith("transformer"):
            if k.endswith(".alpha"):
                v = v[0, 0]
            k = k.replace(".alpha", ".weight")
            k = k.replace(".in_proj_weight", ".in_proj.weight")
            model[k] = v

    # Only export the first 8 slices of the depformer (main).
    n_q_main = 8
    print(f"only exporting the first {n_q_main}/{n_q} depformer layers") 
    for idx in range(n_q_main):
        base = f"depformer.slices.{idx}."
        model[base + "linear_in.weight"] = tch_model[f"depformer_in.{idx}.weight"]
        model[base + "linear_out.weight"] = tch_model[f"linears.{idx}.weight"]
        if idx == 0:
            model[base + "emb.weight"] = tch_model["depformer_text_emb.weight"]
        else:
            model[base + "emb.weight"] = tch_model[f"depformer_emb.{idx-1}.weight"]

        for layer_idx in range(6):
            layer = base + f"transformer.layers.{layer_idx}."
            # WARNING: note that this uses in_proj_weight vs out_proj.weight
            model[layer + "self_attn.in_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"]
                .chunk(n_q)[idx]
                .clone()
            )
            model[layer + "self_attn.out_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"]
                .chunk(n_q)[idx]
                .clone()
            )
            model[layer + "norm1.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.norm1.alpha"
            ][0, 0].clone()
            model[layer + "norm2.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.norm2.alpha"
            ][0, 0].clone()
            model[layer + "gating.linear_in.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{idx}.linear_in.weight"
            ]
            model[layer + "gating.linear_out.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{idx}.linear_out.weight"
            ]

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help="the pytorch checkpoint to import")
    parser.add_argument('out', type=str, help="the mlx safetensors file to generate")
    parser.add_argument('-s', '--silent', action='store_true', help="Only prints the checkpoint name")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    if not out_path.exists():
        import_model(ckpt_path, out_path, silent=args.silent)
    print(out_path)

if __name__ == "__main__":
    main()

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file


def import_model(
    in_path: Path,
    out_path: Path,
    silent: bool = False,
    max_out_n_q: int | None = None,
) -> None:
    if in_path.suffix == ".safetensors":
        tch_model = load_file(in_path)
    else:
        pkg = torch.load(in_path, map_location=torch.device("cpu"))
        tch_model = pkg["fsdp_best_state"]["model"]

    in_n_q: int | None = None
    for idx in range(999):
        name = f"emb.{idx}.weight"
        if name not in tch_model:
            in_n_q = idx
            break
    out_n_q: int | None = None
    for idx in range(999):
        name = f"depformer_in.{idx}.weight"
        if name not in tch_model:
            out_n_q = idx
            break
    assert in_n_q is not None
    assert out_n_q is not None
    if not silent:
        print(f"in_n_q: {in_n_q}, out_n_q: {out_n_q}")

    model = {}
    for name in ["text_emb.weight", "text_linear.weight"]:
        model[name] = tch_model[name]
    model["out_norm.weight"] = tch_model["out_norm.alpha"][0, 0]
    for idx in range(in_n_q):
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
    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        print(f"only exporting the first {exported_out_n_q} depformer layers")
    else:
        exported_out_n_q = out_n_q
    for idx in range(exported_out_n_q):
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
                .chunk(out_n_q)[idx]
                .clone()
            )
            model[layer + "self_attn.out_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"]
                .chunk(out_n_q)[idx]
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
    if "condition_provider.conditioners.description.embed.weight" in tch_model:
        e = tch_model["condition_provider.conditioners.description.embed.weight"]
        w = tch_model["condition_provider.conditioners.description.output_proj.weight"]
        # 4 is very_good
        e = e[4:5] @ w.T
        print(f"adding the very_good conditioning {e.shape} to {model['text_emb.weight'].shape}")
        model["text_emb.weight"] += e

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="the pytorch checkpoint to import")
    parser.add_argument("out", type=str, help="the mlx safetensors file to generate")
    parser.add_argument(
        "-s", "--silent", action="store_true", help="only prints the checkpoint name"
    )
    parser.add_argument(
        "--max-out-n-q",
        type=int,
        help="limit the number of depformer layers that are exported",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    if not out_path.exists():
        import_model(
            ckpt_path, out_path, silent=args.silent, max_out_n_q=args.max_out_n_q
        )
    print(out_path)


if __name__ == "__main__":
    main()

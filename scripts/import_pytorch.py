# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from safetensors.torch import save_file

import torch

DEPFORMER_LAYERS = 6


def import_model(
    in_path: Path,
    out_path: Path,
    max_out_n_q: int | None = None,
) -> None:
    pkg = torch.load(in_path, map_location=torch.device("cpu"))
    model = pkg["fsdp_best_state"]["model"]

    in_n_q: int | None = None
    for idx in range(999):
        name = f"emb.{idx}.weight"
        if name not in model:
            in_n_q = idx
            break

    out_n_q: int | None = None
    for idx in range(999):
        name = f"depformer_in.{idx}.weight"
        if name not in model:
            out_n_q = idx
            break
    assert in_n_q is not None
    assert out_n_q is not None
    print(f"in_n_q: {in_n_q}, out_n_q: {out_n_q}")
    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        print(f"only exporting the first {exported_out_n_q} depformer layers")
    else:
        exported_out_n_q = out_n_q

    if exported_out_n_q > 0:
        for idx in range(DEPFORMER_LAYERS):
            in_proj_key = f"depformer.layers.{idx}.self_attn.in_proj_weight"
            in_proj = model[in_proj_key]
            model[in_proj_key] = in_proj[: in_proj.shape[0] // 2]
            out_proj_key = f"depformer.layers.{idx}.self_attn.out_proj.weight"
            out_proj = model[out_proj_key]
            model[out_proj_key] = out_proj[: out_proj.shape[0] // 2]

        # For mimi inference, we trim the depformer layer that are unused.
        for dep_idx in range(exported_out_n_q - 1, in_n_q - 1):
            del model[f"depformer_emb.{dep_idx}.weight"]
        for dep_idx in range(exported_out_n_q, in_n_q):
            del model[f"linears.{dep_idx}.weight"]
            del model[f"depformer_in.{dep_idx}.weight"]
            for idx in range(DEPFORMER_LAYERS):
                del model[f"depformer.layers.{idx}.gating.{dep_idx}.linear_in.weight"]
                del model[f"depformer.layers.{idx}.gating.{dep_idx}.linear_out.weight"]

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser(
        prog="moshi_import", description="Imports moshi checkpoints"
    )
    parser.add_argument("checkpoint", help="The checkpoint to be imported.")
    parser.add_argument("out", help="The safetensors out file.")
    parser.add_argument(
        "--max-out-n-q",
        type=int,
        help="limit the number of depformer layers that are exported",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    if out_path.exists():
        print("file already exists")
    else:
        import_model(args.checkpoint, out_path, args.max_out_n_q)
    print(out_path)


if __name__ == "__main__":
    main()

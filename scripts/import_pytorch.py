import argparse
import os
from pathlib import Path
from safetensors.torch import save_file

import torch

USER = os.environ["USER"]
OUT_FOLDER = Path(f"/lustre/scwpod02/client/kyutai/{USER}/tmp/mimi")
XP_FOLDER = "/lustre/scwpod02/client/kyutai/{user}/mimi_exp/xps"

DEPFORMER_LAYERS = 6


def import_model(in_path: Path, out_path: Path, silent: bool = False) -> None:
    pkg = torch.load(in_path, map_location=torch.device("cpu"))
    model = pkg["fsdp_best_state"]["model"]
    # For mimi inference, we trim the depformer layer that are unused.
    for idx in range(DEPFORMER_LAYERS):
        in_proj_key = f"depformer.layers.{idx}.self_attn.in_proj_weight"
        in_proj = model[in_proj_key]
        model[in_proj_key] = in_proj[: in_proj.shape[0] // 2]
        out_proj_key = f"depformer.layers.{idx}.self_attn.out_proj.weight"
        out_proj = model[out_proj_key]
        model[out_proj_key] = out_proj[: out_proj.shape[0] // 2]

    for dep_idx in range(7, 15):
        del model[f"depformer_emb.{dep_idx}.weight"]
    for dep_idx in range(8, 16):
        del model[f"linears.{dep_idx}.weight"]
        del model[f"depformer_in.{dep_idx}.weight"]
        for idx in range(DEPFORMER_LAYERS):
            del model[f"depformer.layers.{idx}.gating.{dep_idx}.linear_in.weight"]
            del model[f"depformer.layers.{idx}.gating.{dep_idx}.linear_out.weight"]

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser(
        prog="mimi_import", description="Imports mimi checkpoints"
    )
    parser.add_argument("sig", help="Signature of the xp.")
    parser.add_argument("-e", "--epoch", type=int, help="Epoch to load.")
    parser.add_argument(
        "-u",
        "--user",
        default=USER,
        help="User in which to go look for the checkpoint.",
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Only prints the checkpoint name"
    )
    args = parser.parse_args()

    xp_folder = Path(XP_FOLDER.format(user=args.user))
    if args.epoch is None:
        ckpt_name = "checkpoint.th"
        out_name = f"mimi_{args.sig}.safetensors"
    else:
        ckpt_name = f"checkpoint_{args.epoch}.th"
        out_name = f"mimi_{args.sig}@{args.epoch}.safetensors"
    ckpt_path = xp_folder / args.sig / ckpt_name
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = OUT_FOLDER / out_name
    if out_path.exists():
        print("file already exists")
    else:
        import_model(ckpt_path, out_path, silent=args.silent)
    print(out_path)


if __name__ == "__main__":
    main()

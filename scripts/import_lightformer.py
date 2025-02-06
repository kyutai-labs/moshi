# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Import Moshi model, in particular with support for a 'light' depth transformer
with low rank embeddings and weight sharing for some codebooks."""

import argparse
from pathlib import Path
from safetensors.torch import save_file

import omegaconf
import torch


def import_model(
    in_path: str,
    out_path: str,
) -> None:
    pkg = torch.load(in_path, map_location=torch.device("cpu"))
    if 'xp.cfg' in pkg:
        cfg = pkg['xp.cfg']
    else:
        cfg = omegaconf.OmegaConf.load(Path(in_path).parent / '.hydra/config.yaml')

    model = pkg["fsdp_best_state"]["model"]

    # Asumming same size of both streams n_q.
    in_n_q = cfg.compression_model_n_q * 2
    out_n_q = cfg.compression_model_n_q
    print(f"in_n_q: {in_n_q}, out_n_q: {out_n_q}")
    schedule = cfg.transformer_lm.get('depformer_weights_per_step_schedule', None)
    if schedule is None:
        schedule = list(range(in_n_q))

    num_weights = max(schedule) + 1
    schedule = schedule[:out_n_q]
    kept_weights = max(schedule) + 1
    print(f"Number of dep weights: {num_weights}, keeping {kept_weights}")

    for idx in range(cfg.transformer_lm.depformer_num_layers):
        in_proj_key = f"depformer.layers.{idx}.self_attn.in_proj_weight"
        in_proj = model[in_proj_key]
        in_proj = in_proj.view(num_weights, -1, *in_proj.shape[1:])
        model[in_proj_key] = in_proj[:kept_weights].view(-1, *in_proj.shape[2:]).contiguous()
        out_proj_key = f"depformer.layers.{idx}.self_attn.out_proj.weight"
        out_proj = model[out_proj_key]
        out_proj = out_proj.view(num_weights, -1, *out_proj.shape[1:])
        model[out_proj_key] = out_proj[:kept_weights].view(-1, *out_proj.shape[2:]).contiguous()

    # For mimi inference, we trim the depformer layer that are unused.
    for dep_idx in range(out_n_q - 1, in_n_q - 1):
        del model[f"depformer_emb.{dep_idx}.weight"]
        if cfg.transformer_lm.get('depformer_low_rank_embeddings'):
            del model[f"depformer_emb.{dep_idx}.low_rank.weight"]
    for dep_idx in range(out_n_q, in_n_q):
        del model[f"linears.{dep_idx}.weight"]
    for real_idx in range(kept_weights, num_weights):
        model.pop(f"depformer_in.{real_idx}.weight")
        for idx in range(cfg.transformer_lm.depformer_num_layers):
            model.pop(f"depformer.layers.{idx}.gating.{real_idx}.linear_in.weight")
            model.pop(f"depformer.layers.{idx}.gating.{real_idx}.linear_out.weight")

    schedule = schedule[:out_n_q]

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser(
        prog="moshi_import", description="Imports moshi checkpoints"
    )
    parser.add_argument("checkpoint", help="The checkpoint to be imported.")
    parser.add_argument("out", help="The safetensors out file.")
    args = parser.parse_args()

    out_path = Path(args.out)

    if out_path.exists():
        print("file already exists")
    else:
        import_model(args.checkpoint, out_path)
    print(out_path)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from safetensors.torch import save_file

import torch


def import_model(
    in_path: Path,
    max_out_n_q: int | None,
    out_path: Path,
    weights_per_step_schedule: list[int] | None = None,
) -> None:
    pkg = torch.load(in_path, map_location=torch.device("cpu"), weights_only=False)
    tch_model = pkg["fsdp_best_state"]["model"]
    for k, v in sorted(tch_model.items()):
        print(k, v.shape, v.dtype)

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

    if weights_per_step_schedule is not None:
        if len(weights_per_step_schedule) != out_n_q:
            raise ValueError("inconsistent weights_per_step_schedule", len(weights_per_step_schedule), out_n_q)

    depformer_layers: int | None = None
    for idx in range(999):
        if f"depformer.layers.{idx}.self_attn.in_proj_weight" not in tch_model:
            depformer_layers = idx
            break
    assert depformer_layers is not None
    print(f"depformer layers: {depformer_layers}")

    model = {}
    for name in ["text_emb.weight", "text_linear.weight", "out_norm.alpha"]:
        model[name] = tch_model[name]
    for name in tch_model.keys():
        if name.startswith("condition_provider.conditioners"):
            model[name] = tch_model[name]
    for idx in range(in_n_q):
        name = f"emb.{idx}.weight"
        model[name] = tch_model[name]

    for k, v in sorted(tch_model.items()):
        if k.startswith("transformer") or k.startswith("extra_heads"):
            model[k] = v

    # Only export the first slices of the depformer (main).
    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        print(f"only exporting the first {exported_out_n_q} depformer layers")
    else:
        print(f"exporting all {out_n_q} depformer layers")
        exported_out_n_q = out_n_q

    max_df_steps = out_n_q
    if weights_per_step_schedule is not None:
        max_df_steps = max(weights_per_step_schedule) + 1

    for idx in range(exported_out_n_q):
        if weights_per_step_schedule is not None:
            tch_idx = weights_per_step_schedule[idx]
        else:
            tch_idx = idx
        base = f"depformer.{idx}."
        model[base + "linear_in.weight"] = tch_model[f"depformer_in.{tch_idx}.weight"].clone()
        model[base + "linear_out.weight"] = tch_model[f"linears.{idx}.weight"]
        if idx == 0:
            model[base + "emb.weight"] = tch_model["depformer_text_emb.weight"]
            if "depformer_text_emb.low_rank.weight" in tch_model:
                model[base + "emb.low_rank.weight"] = tch_model["depformer_text_emb.low_rank.weight"].clone()
        else:
            model[base + "emb.weight"] = tch_model[f"depformer_emb.{idx-1}.weight"].clone()
            if f"depformer_emb.{idx-1}.low_rank.weight" in tch_model:
                model[base + "emb.low_rank.weight"] = tch_model[f"depformer_emb.{idx-1}.low_rank.weight"].clone()

        for layer_idx in range(depformer_layers):
            layer = base + f"transformer.layers.{layer_idx}."
            # WARNING: note that this uses in_proj_weight vs out_proj.weight
            model[layer + "self_attn.in_proj_weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"]
                .chunk(max_df_steps)[tch_idx]
                .clone()
            )
            model[layer + "self_attn.out_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"]
                .chunk(max_df_steps)[tch_idx]
                .clone()
            )
            model[layer + "norm1.alpha"] = tch_model[
                f"depformer.layers.{layer_idx}.norm1.alpha"
            ].clone()
            model[layer + "norm2.alpha"] = tch_model[
                f"depformer.layers.{layer_idx}.norm2.alpha"
            ].clone()
            model[layer + "gating.linear_in.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_in.weight"
            ].clone()
            model[layer + "gating.linear_out.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_out.weight"
            ].clone()

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
    parser.add_argument("--wpss", type=str, help="weights per step schedule config")
    args = parser.parse_args()

    wpss = None
    if args.wpss is not None:
        if args.wpss == "hibiki-2b":
            wpss = [0, 1, 2, 3, 4, 5, 6, 7] + [8] * 8 + [9] * 16
        elif args.wpss == "secret":
            wpss = [0, 1, 2, 3, 4, 5, 6, 7] + [8] * 8 + [9] * 8 + [10] * 8
        else:
            raise ValueError(f"unknown wpss {args.wpss}")

    out_path = Path(args.out)

    if out_path.exists():
        print("file already exists")
    else:
        import_model(args.checkpoint, args.max_out_n_q, out_path, wpss)
    print(out_path)


if __name__ == "__main__":
    main()

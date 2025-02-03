import argparse
from pathlib import Path
from safetensors.torch import save_file

import torch


def import_model(in_path: Path, max_out_n_q, out_path: Path) -> None:
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
        if k.startswith("transformer"):
            model[k] = v

    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        print(f"only exporting the first {exported_out_n_q} depformer layers")
    else:
        print(f"exporting all {out_n_q} depformer layers")
        exported_out_n_q = out_n_q

    for idx in range(exported_out_n_q):
        base = f"depformer.{idx}."
        model[base + "linear_in.weight"] = tch_model[f"depformer_in.{idx}.weight"]
        model[base + "linear_out.weight"] = tch_model[f"linears.{idx}.weight"]
        if idx == 0:
            model[base + "emb.weight"] = tch_model["depformer_text_emb.weight"]
        else:
            model[base + "emb.weight"] = tch_model[f"depformer_emb.{idx-1}.weight"]

        for layer_idx in range(6):
            layer = base + f"transformer.layers.{layer_idx}."
            # WARNING: note that this uses in_proj_weight vs out_proj.weight
            model[layer + "self_attn.in_proj_weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"]
                .chunk(out_n_q)[idx]
                .clone()
            )
            model[layer + "self_attn.out_proj.weight"] = (
                tch_model[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"]
                .chunk(out_n_q)[idx]
                .clone()
            )
            model[layer + "norm1.alpha"] = tch_model[
                f"depformer.layers.{layer_idx}.norm1.alpha"
            ].clone()
            model[layer + "norm2.alpha"] = tch_model[
                f"depformer.layers.{layer_idx}.norm2.alpha"
            ].clone()
            model[layer + "gating.linear_in.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{idx}.linear_in.weight"
            ]
            model[layer + "gating.linear_out.weight"] = tch_model[
                f"depformer.layers.{layer_idx}.gating.{idx}.linear_out.weight"
            ]

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
        import_model(args.checkpoint, args.max_out_n_q, out_path)
    print(out_path)


if __name__ == "__main__":
    main()

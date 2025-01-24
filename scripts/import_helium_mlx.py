# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download


def import_model(in_path: Path, out_path: Path, silent: bool = False) -> None:
    with safe_open(in_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    model = {
        "text_emb.weight": tensors["model.embed_tokens.weight"],
        "text_linear.weight": tensors["lm_head.weight"],
        "out_norm.weight": tensors["model.norm.weight"],
    }
    n_layers = -1
    for key in tensors.keys():
        if key.startswith("model.layers."):
            layer_idx = int(key.split(".")[2])
            n_layers = max(layer_idx, n_layers)
    n_layers += 1
    if not silent:
        print(f"found {n_layers} layers")
    for layer_idx in range(n_layers):
        dst_prefix = f"transformer.layers.{layer_idx}."
        src_prefix = f"model.layers.{layer_idx}."
        _model = {
            "norm1.weight": "input_layernorm.weight",
            "norm2.weight": "post_attention_layernorm.weight",
            "self_attn.out_proj.weight": "self_attn.o_proj.weight",
            "gating.linear_out.weight": "mlp.down_proj.weight",
        }
        for dst, src in _model.items():
            model[dst_prefix + dst] = tensors[src_prefix + src]
        gate_proj = tensors[src_prefix + "mlp.gate_proj.weight"]
        up_proj = tensors[src_prefix + "mlp.up_proj.weight"]
        linear_in = torch.cat([gate_proj, up_proj], dim=0)
        model[dst_prefix + "gating.linear_in.weight"] = linear_in
        q = tensors[src_prefix + "self_attn.q_proj.weight"]
        k = tensors[src_prefix + "self_attn.k_proj.weight"]
        v = tensors[src_prefix + "self_attn.v_proj.weight"]
        in_proj = torch.cat([q, k, v], dim=0)
        model[dst_prefix + "self_attn.in_proj.weight"] = in_proj

    save_file(model, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="kyutai/helium-1-preview-2b",
        help="the transformers checkpoint to import",
    )
    parser.add_argument("--out", type=str, help="the mlx safetensors file to generate")
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Only prints the checkpoint name"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = hf_hub_download(
            repo_id=args.checkpoint, filename="model.safetensors"
        )
    out_path = Path(args.out)
    if not out_path.exists():
        import_model(ckpt_path, out_path, silent=args.silent)
    print(out_path)


if __name__ == "__main__":
    main()

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Import Moshi model, in particular with support for a 'light' depth transformer
with low rank embeddings and weight sharing for some codebooks."""

import argparse
import json
from pathlib import Path
import typing as tp

import omegaconf
from safetensors.torch import save_file
import torch


def import_model(
    args,
) -> None:
    args.out_folder.mkdir(exist_ok=True, parents=True)
    out_config = args.out_folder / 'config.json'
    out_file = args.out_folder / 'checkpoint.safetensors'
    pkg = torch.load(args.checkpoint, map_location=torch.device("cpu"), weights_only=False)
    if 'xp.cfg' in pkg:
        cfg = pkg['xp.cfg']
    else:
        cfg = omegaconf.OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')

    model = pkg["fsdp_best_state"]["model"]

    # Asumming same size of both streams n_q.
    n_q = cfg.compression_model_n_q
    if cfg.tokens.multistream:
        n_q *= 2

    in_n_q = n_q
    out_n_q = args.out_n_q or n_q
    print(f"in_n_q: {in_n_q}, out_n_q: {out_n_q}")

    keys = [
        'dim', 'text_card', 'existing_text_padding_id', 'num_heads', 'num_layers', 'hidden_scale', 'causal',
        'layer_scale', 'context', 'max_period', 'gating', 'norm', 'positional_embedding',
        'depformer_dim', 'depformer_num_heads', 'depformer_num_layers', 'depformer_dim_feedforward',
        'depformer_layer_scale', 'depformer_multi_linear',
        'depformer_max_period', 'depformer_gating', 'depformer_pos_emb', 'depformer_weights_per_step',
        'depformer_low_rank_embeddings', 'demux_second_stream',
        'text_card_out']
    config: dict[str, tp.Any] = {}
    config['card'] = 2048
    config['n_q'] = in_n_q
    config['dep_q'] = out_n_q
    tr_args = omegaconf.OmegaConf.to_object(cfg.transformer_lm)
    config['delays'] = tr_args['delays']
    if len(config['delays']) < out_n_q + 1:
        config['delays'] = config['delays'] + [config['delays'][-1]] * (out_n_q + 1 - len(config['delays']))
    for key in keys:
        if key in cfg.transformer_lm:
            config[key] = tr_args[key]
        else:
            print(f"Missing config key {key}")
    if config['norm'].startswith('real_'):
        config['norm'] = config['norm'].removeprefix('real_')
    config['conditioners'] = omegaconf.OmegaConf.to_object(cfg.conditioners)
    config['fuser'] = omegaconf.OmegaConf.to_object(cfg.fuser)
    config['fuser'].pop('streaming_sum', None)
    config['cross_attention'] = bool(config['fuser'].get('cross'))

    if hasattr(cfg, 'interleaver') and cfg.interleaver.variant == 'tts_delay':
        kw_interleaver = dict(cfg.interleaver)
        kw_interleaver.update(cfg.interleaver.tts_delay)
        config['tts_config'] = {
            'audio_delay': cfg.interleaver.audio_delay,
            'second_stream_ahead': kw_interleaver.get('second_stream_ahead', 0),
        }

    config['model_id'] = {}
    if args.sig is not None:
        config['model_id']['sig'] = args.sig
    if args.epoch is not None:
        config['model_id']['epoch'] = args.epoch

    schedule = cfg.transformer_lm.get('depformer_weights_per_step_schedule', None)
    has_schedule = True
    if schedule is None:
        has_schedule = False
        schedule = list(range(in_n_q))
    num_weights = max(schedule) + 1
    schedule = schedule[:out_n_q]
    if has_schedule:
        config['depformer_weights_per_step_schedule'] = schedule

    if args.extra_config:
        extra = json.loads(args.extra_config.read_text())
        config.update(extra)
    out_config.write_text(json.dumps(config, indent=2))

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

    save_file(model, out_file)


def main():
    parser = argparse.ArgumentParser(
        prog="moshi_import", description="Imports moshi checkpoints"
    )
    parser.add_argument("--out_n_q", type=int,
                        help="Number of codebooks to keep in the Depth Transformer.")
    parser.add_argument("--extra_config", type=Path, help="Extra config to add to the json.")
    parser.add_argument("--sig", help="Signature of the original XP for tracability.")
    parser.add_argument("--epoch", type=int, help="Epoch of the original XP.")
    parser.add_argument("checkpoint", type=Path, help="The checkpoint to be imported.")
    parser.add_argument("out_folder", type=Path, help=".")
    args = parser.parse_args()

    print(f"Saving to folder {args.out_folder}")
    import_model(args)


if __name__ == "__main__":
    main()

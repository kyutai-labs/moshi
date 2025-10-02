# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Import Mimi codecs."""

import argparse
from functools import partial
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
    out_config = args.out_folder / 'mimi_config.json'
    out_file = args.out_folder / 'mimi.safetensors'
    pkg = torch.load(args.checkpoint, map_location=torch.device("cpu"), weights_only=False)
    if 'xp.cfg' in pkg:
        cfg = pkg['xp.cfg']
    else:
        cfg = omegaconf.OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')

    model = pkg["best_state"]["model"]
    assert model
    for key, value in dict(model).items():
        if key.startswith('wavlm_'):
            model.pop(key)
        elif key.endswith('_v'):
            base = key[:-2]
            model.pop(key)
            other = model.pop(base + '_g')
            model[base] = torch._weight_norm(value, other, dim=0)

    config = {
        'channels': cfg.channels,
        'sample_rate': cfg.sample_rate,
        'frame_rate': cfg.encodec.frame_rate
    }
    to_container = partial(omegaconf.OmegaConf.to_container, resolve=True)
    config['seanet'] = to_container(cfg.seanet)
    config['seanet'].pop('lstm')
    config['seanet'].pop('encoder')
    config['seanet'].pop('decoder')
    config['seanet']['norm'] = 'none'
    config['quantizer'] = to_container(cfg.rvq)
    keep = [
        'dimension', 'n_q', 'bins'
    ]
    for param in list(config['quantizer']):
        if param not in keep:
            config['quantizer'].pop(param)
    config['quantizer']['input_dimension'] = config['seanet']['dimension']
    config['quantizer']['output_dimension'] = config['seanet']['dimension']

    config['transformer'] = to_container(cfg.transformer)
    config['transformer']['d_model'] = config['seanet']['dimension']
    config['transformer']['input_dimension'] = config['seanet']['dimension']
    config['transformer']['output_dimensions'] = [config['seanet']['dimension']]
    assert config['transformer'].pop('encoder') == {}
    assert config['transformer'].pop('decoder') == {}
    assert config['transformer'].pop('use')
    ignore = ['weight_decay', 'lr', 'betas']
    for param in ignore:
        config['transformer'].pop(param)
    scale = config['transformer'].pop('hidden_scale')
    config['transformer']['dim_feedforward'] = int(config['transformer']['d_model'] * scale)
    config['transformer']['conv_layout'] = True

    if args.extra_config:
        extra = json.loads(args.extra_config.read_text())
        config.update(extra)
    out_config.write_text(json.dumps(config, indent=2))
    save_file(model, out_file)


def main():
    parser = argparse.ArgumentParser(
        prog="import_mimi_pytorch", description="Imports moshi checkpoints"
    )
    parser.add_argument("--extra_config", type=Path, help="Extra config to add to the json.")
    parser.add_argument("checkpoint", type=Path, help="The checkpoint to be imported.")
    parser.add_argument("out_folder", type=Path, help=".")
    args = parser.parse_args()

    print(f"Saving to folder {args.out_folder}")
    import_model(args)


if __name__ == "__main__":
    main()

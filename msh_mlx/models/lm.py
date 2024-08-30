# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..modules.transformer import Config as TransformerConfig
from ..modules.transformer import Transformer

@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int

@dataclass
class Config:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int


class DepFormerSlice(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, main_transformer_dim: int, cfg: TransformerConfig):
        super().__init__()

        dim = cfg.d_model
        self.emb = nn.Embedding(in_vocab_size, dim)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)

    def __call__(self, xs: mx.array) -> mx.array:
        return xs

class DepFormer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.slices = []
        for slice_idx in range(cfg.depformer.num_slices):
            in_vs = cfg.text_in_vocab_size if slice_idx == 0 else cfg.audio_vocab_size
            slice = DepFormerSlice(
                in_vs,
                cfg.audio_vocab_size - 1,
                main_transformer_dim=cfg.transformer.d_model,
                cfg=cfg.depformer.transformer,
            )
            self.slices.append(slice)


    def __call__(self, xs: mx.array) -> mx.array:
        # TODO
        return xs

class Lm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        dim = cfg.transformer.d_model
        self.transformer = Transformer(cfg.transformer)
        self.depformer = DepFormer(cfg)
        self.text_emb = nn.Embedding(cfg.text_in_vocab_size, dim)

        if cfg.transformer.norm == "layer_norm":
            self.out_norm = nn.LayerNorm(dim, 1e-5)
        elif cfg.transformer.norm == "rms_norm":
            self.out_norm = nn.RMSNorm(dim, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.transformer.norm}")

        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [nn.Embedding(cfg.audio_vocab_size, dim) for _i in range(cfg.audio_codebooks)]


    def __call__(self, xs: mx.array) -> mx.array:
        # TODO
        return xs

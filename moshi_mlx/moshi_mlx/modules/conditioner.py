# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
Conditioners
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TensorConditionerConfig:
    dim: int


class TensorConditioner(nn.Module):
    def __init__(self, output_dim: int, cfg: TensorConditionerConfig):
        super().__init__()

        self.output_proj = nn.Linear(cfg.dim, output_dim, bias=False)
        self.learnt_padding = mx.zeros((1, 1, output_dim))


@dataclass
class LutConditionerConfig:
    n_bins: int
    dim: int
    tokenizer: str
    possible_values: dict[str, int]


class LutConditioner(nn.Module):
    def __init__(self, output_dim: int, cfg: LutConditionerConfig):
        super().__init__()

        if cfg.tokenizer != "noop":
            raise ValueError(f"unsupported tokenizer {cfg.tokenizer}")

        self.embed = nn.Embedding(cfg.n_bins + 1, cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, output_dim, bias=False)
        self.learnt_padding = mx.zeros((1, 1, output_dim))
        self.possible_values = { v: i for i, v in enumerate(cfg.possible_values) }

    def condition(self, value: str) -> mx.array:
        idx = self.possible_values.get(value, None)
        if idx is None:
            raise ValueError(f"unknown value {value}, possible-values: {self.possible_values}")
        idx = mx.array([idx])
        return self.output_proj(self.embed(idx))

@dataclass
class ConditionTensor:
    tensor: mx.array

class ConditionProvider(nn.Module):
    def __init__(self, output_dim: int, cfg: dict):
        self.conditioners = {}
        for name, c in cfg.items():
            if isinstance(c, LutConditionerConfig):
                cond = LutConditioner(output_dim, c)
            elif isinstance(c, TensorConditionerConfig):
                cond = TensorConditioner(output_dim, c)
            else:
                raise ValueError(f"unsupported config type {type(c)}")
            self.conditioners[name] = cond

    def condition_tensor(self, name: str, value: str) -> ConditionTensor:
        if name not in self.conditioners:
            raise ValueError(f"unsupported conditioner {name}")
        tensor = self.conditioners[name].condition(value)
        return ConditionTensor(tensor)

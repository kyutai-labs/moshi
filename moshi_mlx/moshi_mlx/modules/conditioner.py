# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
Conditioners
"""

from dataclasses import dataclass, field
import typing as tp

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class TensorCondition:
    """Looks quite similar to ConditionType, but represents the input to TensorConditioners.
    `tensor` should be [B | 1, T, D], and `mask` should be `[B | 1, T]`.
    """
    tensor: mx.array
    mask: mx.array

    @staticmethod
    def from_tensor(tensor: mx.array):
        B, T, _ = tensor.shape
        mask = mx.ones((B, T), dtype=mx.uint8)
        return TensorCondition(tensor, mask)

    @staticmethod
    def cat(conditions: tp.Sequence['TensorCondition']) -> 'TensorCondition':
        assert conditions, "Cannot cat empty list."
        ref_tensor = conditions[0].tensor
        B, _, D = ref_tensor.shape
        assert B == 1
        B = len(conditions)
        T = max(condition.tensor.shape[1] for condition in conditions)
        mask = mx.zeros((B, T), dtype=mx.uint8)
        tensor = mx.zeros((B, T, D), dtype=ref_tensor.dtype)
        for b, condition in enumerate(conditions):
            tensor[b, :condition.tensor.shape[1], :] = condition.tensor[0]
            mask[b, :condition.mask.shape[1]] = condition.mask[0]
        return TensorCondition(tensor, mask)


@dataclass
class ConditionAttributes:
    """Standard class for representing the set of potential inputs to the conditioners.
    Typically, `audiocraft.data.audio_dataset.SegmentInfo` will convert
    to this class to make conditioning agnostic to the type of dataset.

    There are two kinds of conditionings: text (or None), or raw torch tensors (with a mask).

    """
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    tensor: tp.Dict[str, TensorCondition] = field(default_factory=dict)

    @property
    def text_attributes(self) -> tp.Iterable[str]:
        return self.text.keys()

    @property
    def tensor_attributes(self) -> tp.Iterable[str]:
        return self.text.keys()

    @staticmethod
    def condition_types() -> tp.FrozenSet[str]:
        return frozenset(["text", "tensor"])

    def copy(self) -> 'ConditionAttributes':
        return ConditionAttributes(dict(self.text), dict(self.tensor))



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

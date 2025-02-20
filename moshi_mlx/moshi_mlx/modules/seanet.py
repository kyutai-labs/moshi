# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from .conv import StreamableConv1d, StreamableConvTranspose1d

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SeanetConfig:
    dimension: int
    channels: int
    causal: bool
    nfilters: int
    nresidual_layers: int
    ratios: list[int]
    ksize: int
    residual_ksize: int
    last_ksize: int
    dilation_base: int
    pad_mode: str
    true_skip: bool
    compress: int

class SeanetResnetBlock(nn.Module):
    pass

    def reset_state(self):
        pass

class EncoderLayer(nn.Module):
    def __init__(self, cfg: SeanetConfig, ratio: int, mult: int):
        pass

    def reset_state(self):
        pass

class SeanetEncoder(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        mult = 1
        self.init_conv1d = StreamableConv1d(
            in_channels=cfg.channels,
            out_channels=mult * cfg.nfilters,
            ksize=cfg.ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )
        layers = []
        for ratio in reversed(cfg.ratios):
            layers.append(EncoderLayer(cfg, ratio=ratio, mult=mult))
            mult *= 2
        self.layers = layers
        self.final_conv1d = StreamableConv1d(
            in_channels=mult * cfg.nfilters,
            out_channels=cfg.dimension,
            ksize=cfg.last_ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self):
        self.init_conv1d.reset_state()
        self.final_conv1d.reset_state()
        for layer in self.layers:
            layer.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d(xs)
        for layer in self.layers:
            xs = layer(xs)
        xs = nn.elu(xs, alpha=1.0)
        return self.final_conv1d(xs)

class DecoderLayer(nn.Module):
    def __init__(self, cfg: SeanetConfig, ratio: int, mult: int):
        self.upsample = StreamableConvTranspose1d(
            in_channels=mult * cfg.nfilters,
            out_channels=mult * cfg.nfilters // 2,
            ksize=ratio * 2,
            stride=ratio,
            groups=1,
            bias=True,
            causal=cfg.causal,
        )
        self.residuals = []

    def reset_state(self):
        self.upsample.reset_state()
        for r in self.residuals:
            r.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.upsample(nn.elu(xs, alpha=1.0))
        for r in self.residuals:
            xs = r(xs)
        return xs

class SeanetDecoder(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        mult = 1 << len(cfg.ratios)
        self.init_conv1d = StreamableConv1d(
            in_channels=cfg.dimension,
            out_channels=mult * cfg.nfilters,
            ksize=cfg.ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )
        layers = []
        for ratio in cfg.ratios:
            layers.append(DecoderLayer(cfg, ratio=ratio, mult=mult))
            mult //= 2
        self.layers = layers
        self.final_conv1d = StreamableConv1d(
            in_channels=cfg.nfilters,
            out_channels=cfg.channels,
            ksize=cfg.last_ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self):
        self.init_conv1d.reset_state()
        self.final_conv1d.reset_state()
        for layer in self.layers:
            layer.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d(xs)
        for layer in self.layers:
            xs = layer(xs)
        xs = nn.elu(xs, alpha=1.0)
        return self.final_conv1d(xs)

class Seanet(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        self.encoder = SeanetEncoder(cfg)
        self.decoder = SeanetDecoder(cfg)

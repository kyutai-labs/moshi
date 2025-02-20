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
    def __init__(self, cfg: SeanetConfig, dim: int, ksizes_and_dilations: list):
        super().__init__()
        block = []
        hidden = dim // cfg.compress
        for i, (ksize, dilation) in enumerate(ksizes_and_dilations):
            in_channels = dim if i == 0 else hidden
            out_channels = dim if i == len(ksizes_and_dilations) - 1 else hidden
            c = StreamableConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=1,
                dilation=dilation,
                groups=1,
                bias=True,
                causal=cfg.causal,
                pad_mode=cfg.pad_mode,
            )
            block.append(c)
        self.block = block

        if cfg.true_skip:
            self.shortcut = None
        else:
            self.shortcut = StreamableConv1d(
                in_channels=dim,
                out_channels=dim,
                ksize=1,
                stride=1,
                dilation=1,
                groups=1,
                bias=True,
                causal=cfg.causal,
                pad_mode=cfg.pad_mode,
            )

    def reset_state(self):
        if self.shortcut is not None:
            self.shortcut.reset_state()
        for b in self.block:
            b.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        residual = xs
        for b in self.block:
            xs = b(nn.elu(xs, alpha=1.0))
        # TODO(laurent): we might need some streaming additions below.
        if self.shortcut is None:
            xs = xs + residual
        else:
            xs = xs + self.shortcut(residual)
        return xs

class EncoderLayer(nn.Module):
    def __init__(self, cfg: SeanetConfig, ratio: int, mult: int):
        super().__init__()
        residuals = []
        dilation = 1
        for _ in range(cfg.nresidual_layers):
            b = SeanetResnetBlock(
                cfg,
                dim=mult * cfg.nfilters,
                ksizes_and_dilations=[(cfg.residual_ksize, dilation), (1, 1)],
            )
            residuals.append(b)
            dilation *= cfg.dilation_base
        self.residuals = residuals
        self.downsample = StreamableConv1d(
            in_channels=mult * cfg.nfilters,
            out_channels=mult * cfg.nfilters * 2,
            ksize=ratio * 2,
            stride=ratio,
            dilation=1,
            groups=1,
            bias=True,
            causal=True,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self):
        self.downsample.reset_state()
        for r in self.residuals:
            r.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        for r in self.residuals:
            xs = r(xs)
        return self.downsample(nn.elu(xs, alpha=1.0))

class SeanetEncoder(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.encoder = SeanetEncoder(cfg)
        self.decoder = SeanetDecoder(cfg)

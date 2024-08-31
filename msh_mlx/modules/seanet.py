# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..modules.conv import StreamableConv1d

@dataclass
class SeaNetConfig:
    dimension: int
    channels: int
    causal: bool
    n_filters: int
    n_residual_layers: int
    ratios: List[int]
    activation: str
    norm: str
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    true_skip: bool
    compress: int
    lstm: int
    disable_norm_outer_blocks: int
    final_activation: Optional[str]


class SeaNetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        k_sizes_and_dilations: List[Tuple[int, int]],
        norm: Optional[str],
        cfg: SeaNetConfig,
    ):
        super().__init__()

        self.block = []
        hidden = dim // cfg.compress
        for idx, (k_size, dilation) in enumerate(k_sizes_and_dilations):
            in_c = hidden if idx else dim
            out_c = dim if idx == len(k_sizes_and_dilations) - 1 else hidden
            b = StreamableConv1d(
                in_c,
                out_c,
                k_size=k_size,
                stride=1,
                dilation=dilation,
                groups=1,
                bias=True,
                causal=cfg.causal,
                norm=norm,
                pad_mode=cfg.pad_mode,
            )
            self.block.append(b)
        self.shortcut = None
        if not cfg.true_skip:
            self.shortcut = StreamableConv1d(
                dim,
                dim,
                k_size=1,
                stride=1,
                dilation=1,
                grous=1,
                bias=True,
                causal=cfg.causal,
                norm=norm,
                pad_mode=cfg.pad_mode,
            )
        if cfg.activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"unsupported activation {cfg.activation}")

class SeaNetEncoder(nn.Module):
    def __init__(self, cfg: SeaNetConfig):
        super().__init__()

        assert not cfg.lstm, "seanet lstm is not supported"
        mult = 1
        init_norm = None if cfg.disable_norm_outer_blocks >= 1 else cfg.norm
        n_blocks = 2 + len(cfg.ratios)

        self.init_conv1d = StreamableConv1d(
            cfg.channels,
            mult * cfg.n_filters,
            k_size=cfg.kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            norm=init_norm,
            pad_mode=cfg.pad_mode,
        )

        for i, ratio in enumerate(cfg.ratios):
            norm = None if cfg.disable_norm_outer_blocks >= i + 2 else cfg.norm
            self.residuals = []
            self.downsamples = []
            for j in range(cfg.n_residual_layers):
                resnet_block = SeaNetResnetBlock(
                    dim=mult * cfg.n_filters,
                    k_sizes_and_dilations=[(cfg.residual_kernel_size, cfg.dilation_base ** j), (1, 1)],
                    norm=norm,
                    cfg=cfg,
                )
                downsample = StreamableConv1d(
                    mult * cfg.n_filters,
                    mult * cfg.n_filters * 2,
                    k_size=ratio * 2,
                    stride=ratio,
                    dilation=1,
                    groups=1,
                    bias=True,
                    causal=True,
                    norm=norm,
                    pad_mode=cfg.pad_mode,
                )
                self.residuals.append(resnet_block)
                self.downsamples.append(downsample)
                mult *= 2
        final_norm = None if cfg.disable_norm_outer_blocks >= n_blocks else cfg.norm
        self.final_conv1d = StreamableConv1d(
            mult * cfg.n_filters,
            cfg.dimension,
            k_size=cfg.last_kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            norm=final_norm,
            pad_mode=cfg.pad_mode,
        )


class SeaNetDecoder(nn.Module):
    def __init__(self, cfg: SeaNetConfig):
        super().__init__()

        assert not cfg.lstm, "seanet lstm is not supported"
        n_blocks = 2 + len(cfg.ratios)
        mult = 1 << len(cfg.ratios)
        init_norm = None if cfg.disable_norm_outer_blocks == n_blocks else cfg.norm
        self.init_conv1d = StreamableConv1d(
            cfg.dimension,
            mult * cfg.n_filters,
            k_size=cfg.kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            norm=init_norm,
            pad_mode=cfg.pad_mode,
        )

        final_norm = None if cfg.disable_norm_outer_blocks >= 1 else cfg.norm
        self.final_conv1d = StreamableConv1d(
            cfg.n_filters,
            cfg.channels,
            k_size=cfg.last_kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            norm=final_norm,
            pad_mode=cfg.pad_mode,
        )




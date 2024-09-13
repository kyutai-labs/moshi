# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from einops import rearrange
import torch
from torch import nn

from .conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    If `causal` is True, the output uses a causal convolution.
    """

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            pad_mode="replicate",
        )
        if not learnt:
            actual_conv = self.conv.conv.conv
            actual_conv.weight.requires_grad_(False)
            actual_conv.weight.data.fill_(1.0 / (2 * stride))

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.conv(x)
        if not self.learnt:
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.convtr = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
        )
        if not learnt:
            actual_convtr = self.convtr.convtr.convtr
            actual_convtr.weight.requires_grad_(False)
            actual_convtr.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.convtr(x)
        if not self.learnt:
            x_for_normalization = torch.ones_like(x[:1])
            normalization = self.convtr(x_for_normalization)
            y = y / normalization
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y

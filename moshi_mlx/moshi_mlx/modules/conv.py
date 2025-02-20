# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import mlx.core as mx
import mlx.nn as nn

class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        nn.Conv1d
        scale = 1 / (in_channels * ksize)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, ksize, in_channels // groups),
        )
        self.bias = None
        if bias:
            self.bias = mx.zeros(out_channels)
        self._padding = padding
        self._groups = groups
        self._stride = stride
        self._dilation = dilation

    def __call__(self, xs: mx.array) -> mx.array:
        # MLX uses NLC whereas pytorch/candle use NCL
        y = mx.conv1d(
            xs.swapaxes(-1, -2),
            self.weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(-1, -2)

class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        nn.Conv1d
        scale = 1 / (in_channels * ksize)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels // groups, ksize, in_channels),
        )
        self.bias = None
        if bias:
            self.bias = mx.zeros(out_channels)
        self._padding = padding
        self._groups = groups
        self._stride = stride

    def __call__(self, xs: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            xs.swapaxes(-1, -2),
            self.weight,
            stride=self._stride,
            padding=self._padding,
            groups=self._groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

class NormConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        self.conv = Conv1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )

    def __call__(self, xs: mx.array) -> mx.array:
        return self.conv(xs)

class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        self.convtr = ConvTranspose1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def __call__(self, xs: mx.array) -> mx.array:
        return self.convtr(xs)

def get_extra_padding_for_conv1d(
    xs: mx.array,
    ksize: int,
    stride: int, 
    padding_total: int,
) -> int:
    l = xs.shape[-1]
    nframes = max(l + padding_total - ksize, 0) / stride + 1.0
    ideal_len = (int(math.ceil(nframes)) - 1) * stride + ksize - padding_total
    return max(0, ideal_len - l)

def unpad1d(xs: mx.array, unpad_l: int, unpad_r: int) -> mx.array:
    left = unpad_l
    right = xs.shape[-1] - unpad_r
    return xs[..., left:right]

# TODO(laurent): add a streaming module abstract class?
class StreamableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        dilation: int,
        groups: int,
        bias: bool,
        causal: bool,
        pad_mode: str,
    ):
        self._causal = causal
        self._pad_mode = pad_mode
        self._ksize = ksize
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self._prev_xs = None
        self._left_pad_applied = False
        self._out_channels = out_channels

    def reset(self):
        self._prev_xs = None
        self._left_pad_applied = False

    def __call__(self, xs: mx.array) -> mx.array:
        b, _, l = xs.shape
        if l == 0:
            return mx.zeros((b, self._out_channels, 0))
        stride = self.conv.conv._stride
        dilation = self.conv.conv._dilation
        ksize = (self._ksize - 1) * dilation + 1
        if not self._left_pad_applied:
            self._left_pad_applied
            padding_total = ksize - stride
            xs = mx.pad(
                xs,
                pad_width=((0, 0), (0, 0), (padding_total, 0)),
                mode=self._pad_mode
            )
        if self._prev_xs is not None:
            xs = mx.concat([self._prev_xs, xs], axis=-1)
        l = xs.shape[-1]
        nframes = max(l + stride - ksize, 0) // stride
        if nframes > 0:
            offset = nframes * stride
            self._prev_xs = xs[..., offset:]
            in_l = (nframes - 1) * stride + ksize
            if in_l > 0:
                xs = xs[..., 0:in_l]
                return self.conv(xs)
            else:
                return mx.zeros((b, self._out_channels, 0))
        else:
            self._prev_xs = xs
            return mx.zeros((b, self._out_channels, 0))

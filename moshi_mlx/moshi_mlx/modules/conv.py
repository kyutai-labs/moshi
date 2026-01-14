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
        bias: bool = True,
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
            groups=self._groups,
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
        bias: bool = True,
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
        self._ksize = ksize
        self._in_channels = in_channels
        self._out_channels = out_channels
        if groups == in_channels and groups == out_channels:
            eye = (
                mx.eye(out_channels)
                .astype(self.weight.dtype)
                .reshape((out_channels, 1, out_channels))
            )
            eye = mx.repeat(eye, repeats=ksize, axis=1)
            self._expanded_weight = mx.repeat(self.weight, repeats=groups, axis=0) * eye
            self._expanded_groups = 1
        elif groups > 1:
            raise ValueError("groups are not supported in ConvTranspose1d")
        else:
            self._expanded_weight = self.weight
            self._expanded_groups = groups

    def update_in_place(self):
        groups = self._groups
        in_channels = self._in_channels
        out_channels = self._out_channels
        ksize = self._ksize
        if groups == in_channels and groups == out_channels:
            eye = (
                mx.eye(out_channels)
                .astype(self.weight.dtype)
                .reshape((out_channels, 1, out_channels))
            )
            eye = mx.repeat(eye, repeats=ksize, axis=1)
            self._expanded_weight = mx.repeat(self.weight, repeats=groups, axis=0) * eye
            self._expanded_groups = 1
        elif groups > 1:
            raise ValueError("groups are not supported in ConvTranspose1d")
        else:
            self._expanded_weight = self.weight
            self._expanded_groups = groups

    def update(self, parameters: dict) -> nn.Module:
        super().update(parameters)
        self.update_in_place()
        return self

    def __call__(self, xs: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            xs.swapaxes(-1, -2),
            self._expanded_weight,
            stride=self._stride,
            padding=self._padding,
            groups=self._expanded_groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(-1, -2)


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
            bias=bias,
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
            bias=bias,
        )

    def __call__(self, xs: mx.array) -> mx.array:
        return self.convtr(xs)


def get_extra_padding_for_conv1d(
    xs: mx.array,
    ksize: int,
    stride: int,
    padding_total: int,
) -> int:
    len_ = xs.shape[-1]
    nframes = max(len_ + padding_total - ksize, 0) / stride + 1.0
    ideal_len = (int(math.ceil(nframes)) - 1) * stride + ksize - padding_total
    return max(0, ideal_len - len_)


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

    def reset_state(self):
        self._prev_xs = None
        self._left_pad_applied = False

    def __call__(self, xs: mx.array) -> mx.array:
        ksize = self._ksize
        ksize = (ksize - 1) * self.conv.conv._dilation + 1
        padding_total = ksize - self.conv.conv._stride
        extra_padding = get_extra_padding_for_conv1d(
            xs,
            ksize=ksize,
            stride=self.conv.conv._stride,
            padding_total=padding_total,
        )
        z = 0, 0
        if self._causal:
            padding_left = padding_total
            padding_right = 0
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
        widths = [z, z, (padding_left, padding_right + extra_padding)]
        pd = mx.pad(xs, pad_width=widths, mode=self._pad_mode)
        return self.conv(pd)

    def step(self, xs: mx.array) -> mx.array:
        b, _, len_ = xs.shape
        if len_ == 0:
            return mx.zeros((b, self._out_channels, 0))
        stride = self.conv.conv._stride
        dilation = self.conv.conv._dilation
        ksize = (self._ksize - 1) * dilation + 1
        if not self._left_pad_applied:
            self._left_pad_applied = True
            padding_total = ksize - stride
            xs = mx.pad(
                xs, pad_width=((0, 0), (0, 0), (padding_total, 0)), mode=self._pad_mode
            )
        if self._prev_xs is not None:
            xs = mx.concat([self._prev_xs, xs], axis=-1)
        len_ = xs.shape[-1]
        nframes = max(len_ + stride - ksize, 0) // stride
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


class StreamableConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int,
        bias: bool,
        causal: bool,
    ):
        self._causal = causal
        self._ksize = ksize
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self._prev_ys = None

    def reset_state(self):
        self._prev_ys = None

    def __call__(self, xs: mx.array) -> mx.array:
        stride = self.convtr.convtr._stride
        padding_total = max(self._ksize - stride, 0)
        xs = self.convtr(xs)
        if self._causal:
            unpad_l = 0
            unpad_r = padding_total
        else:
            unpad_r = padding_total // 2
            unpad_l = padding_total - unpad_r
        return unpad1d(xs, unpad_l=unpad_l, unpad_r=unpad_r)

    def step(self, xs: mx.array) -> mx.array:
        b, _, len_ = xs.shape
        if len_ == 0:
            return mx.zeros((b, self._out_channels, 0))
        stride = self.convtr.convtr._stride
        ys = self.convtr(xs)
        ot = ys.shape[-1]
        if self._prev_ys is not None:
            prev_ys = self._prev_ys
            pt = prev_ys.shape[-1]
            if self.convtr.convtr.bias is not None:
                prev_ys = prev_ys - self.convtr.convtr.bias[None, :, None]
            ys1, ys2 = ys[..., :pt] + prev_ys, ys[..., pt:]
            ys = mx.concat([ys1, ys2], axis=-1)
        invalid_steps = self._ksize - stride
        ys, self._prev_ys = ys[..., : ot - invalid_steps], ys[..., ot - invalid_steps :]
        return ys


class ConvDownsample1d(nn.Module):
    def __init__(self, stride: int, dim: int, causal: bool):
        self.conv = StreamableConv1d(
            in_channels=dim,
            out_channels=dim,
            ksize=2 * stride,
            stride=stride,
            dilation=1,
            groups=1,
            bias=False,
            causal=causal,
            pad_mode="edge",
        )

    def reset_state(self):
        self.conv.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.conv(xs)

    def step(self, xs: mx.array) -> mx.array:
        return self.conv.step(xs)


class ConvTrUpsample1d(nn.Module):
    def __init__(self, stride: int, dim: int, causal: bool):
        self.convtr = StreamableConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            ksize=2 * stride,
            stride=stride,
            groups=dim,  # TODO: hopefully someday this will be fixed.
            bias=False,
            causal=causal,
        )

    def reset_state(self):
        self.convtr.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.convtr(xs)
        return xs

    def step(self, xs: mx.array) -> mx.array:
        xs = self.convtr.step(xs)
        return xs

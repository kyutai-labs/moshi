# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

def get_extra_padding_for_conv1d(
    xs: mx.array,
    kernel_size: int,
    stride: int,
    padding_total: int = 0
) -> int:
    length = xs.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    xs: mx.array,
    kernel_size: int,
    stride: int,
    padding_total: int,
) -> mx.array:
    extra_padding = get_extra_padding_for_conv1d(xs, kernel_size, stride, padding_total)
    return mx.pad(xs, (0, extra_padding))

def pad1d(
    xs: mx.array,
    paddings: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = xs.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            xs = mx.pad(xs, (0, extra_pad))
        padded = mx.pad(xs, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return mx.pad(xs, paddings, mode, value)


def unpad1d(xs: mx.array, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= xs.shape[-1]
    end = xs.shape[-1] - padding_right
    return xs[..., padding_left:end]

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(StreamingConv1d(*args, **kwargs), norm)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            StreamingConvTranspose1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        return x


class StreamableConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self._is_streaming:
            assert self.causal, "streaming is only supported for causal convs"
            padding_to_add = self._streaming_state.get("padding_to_add")
            if padding_to_add is None:
                self._streaming_state["padding_to_add"] = padding_total
                padding_to_add = padding_total
            if padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (padding_to_add, 0), mode=self.pad_mode)
                self._streaming_state["padding_to_add"] = 0
        else:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        return self.conv(x)


class StreamableConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        if not self._is_streaming:
            # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
            # removed at the very end, when keeping only the right length for the output,
            # as removing it here would require also passing the length at the matching layer
            # in the encoder.
            if self.causal:
                # Trim the padding on the right according to the specified ratio
                # if trim_right_ratio = 1.0, trim everything from right
                padding_right = math.ceil(padding_total * self.trim_right_ratio)
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
        return y

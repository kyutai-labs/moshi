# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

class NormConv1d(nn.Module):
    def __init__(self):
        super().__init__()

class NormConvTranspose1d(nn.Module):
    def __init__(self):
        super().__init__()

class StreamableConv1d(nn.Module):
    def __init__(self):
        super().__init__()

class StreamableConvTranspose1d(nn.Module):
    def __init__(self):
        super().__init__()

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


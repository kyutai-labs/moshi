# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import itertools
import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .streaming import StreamingModule, State


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])
M = tp.TypeVar('M', bound=nn.Module)


class TransposedLayerNorm(nn.Module):
    """LayerNorm for [B, C, T] inputs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(**kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(1, 2)


def apply_parametrization_norm(module: M, norm: str = "none") -> M:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(
            nn.Conv1d(*args, **kwargs), norm
        )
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
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        return x


@dataclass
class _StreamingConv1dState(State):
    previous: torch.Tensor
    first: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.previous[:] = torch.where(reset_mask.view(-1, 1, 1), torch.zeros_like(self.previous), self.previous)
        self.first[:] = torch.where(reset_mask, torch.ones_like(self.first), self.first)


class StreamingConv1d(StreamingModule[_StreamingConv1dState]):
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
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ['constant', 'replicate'], pad_mode
        self.pad_mode = pad_mode
        assert causal
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
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

    @property
    def _stride(self) -> int:
        return self.conv.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.conv.conv.dilation[0]
        return (
            self._kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        stride = self._stride
        # Effective kernel size accounting for dilation.
        kernel = self._effective_kernel_size
        param = next(iter(self.parameters()))
        dtype = param.dtype
        device = param.device
        previous = torch.zeros(batch_size, self.conv.conv.in_channels, kernel - stride,
                               dtype=dtype, device=device)
        first = torch.ones(batch_size, device=device, dtype=torch.bool)
        return _StreamingConv1dState(batch_size, device, previous, first)

    def forward(self, x):
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        state = self._streaming_state
        if state is None:
            state = self._init_streaming_state(B)
        TP = state.previous.shape[-1]
        if TP and self.pad_mode == 'replicate':
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state.previous[:] = torch.where(
                state.first.view(-1, 1, 1) & state.exec_mask.view(-1, 1, 1),
                init,
                state.previous)
        if TP:
            x = torch.cat([state.previous, x], dim=-1)
        y = self.conv(x)
        if TP:
            state.previous[:] = torch.where(
                state.exec_mask.view(-1, 1, 1),
                x[..., -TP:],
                state.previous)
            if self.pad_mode == 'replicate':
                state.first = torch.where(
                    state.exec_mask,
                    torch.zeros_like(state.first),
                    state.first,
                )
        return y


@dataclass
class _StreamingConvTr1dState(State):
    partial: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.partial[:] = torch.where(
            reset_mask.view(-1, 1, 1),
            torch.zeros_like(self.partial),
            self.partial)


class StreamingConvTranspose1d(StreamingModule[_StreamingConvTr1dState]):
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
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        super().__init__()
        assert trim_right_ratio == 1.
        assert causal
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

    @property
    def _stride(self) -> int:
        return self.convtr.convtr.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.convtr.convtr.kernel_size[0]

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTr1dState:
        param = next(iter(self.parameters()))
        dtype = param.dtype
        device = param.device
        K = self._kernel_size
        S = self._stride
        partial = torch.zeros(batch_size, self.convtr.convtr.out_channels, K - S,
                              device=device, dtype=dtype)
        return _StreamingConvTr1dState(batch_size, device, partial)

    def forward(self, x):
        B, C, T = x.shape
        K = self._kernel_size
        S = self._stride
        state = self._streaming_state

        y = self.convtr(x)
        if state is None:
            y = unpad1d(y, (0, K - S))
        else:
            PT = state.partial.shape[-1]
            if PT > 0:
                y[..., :PT] += state.partial
                bias = self.convtr.convtr.bias
                for_partial = y[..., -PT:]
                if bias is not None:
                    for_partial -= bias[:, None]
                state.partial[:] = torch.where(
                    state.exec_mask.view(-1, 1, 1),
                    for_partial,
                    state.partial)
                y = y[..., :-PT]
        return y


def test():
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        # Avoid the cuda optimizations that would take place on single precision
        # floats for convolutions.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    kernel_sizes = [1, 3, 4, 8, 15, 16]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chin = 6
    chout = 12

    for kernel, stride in itertools.product(kernel_sizes, strides):
        if stride > kernel:
            continue
        conv = StreamingConv1d(chin, chout, kernel, stride, causal=True).to(device)
        convtr = StreamingConvTranspose1d(chout, chin, kernel, stride, causal=True).to(device)

        for frames in [1, 4, 8, 32, 54, 65, 128]:
            print(f"ksize {kernel} strides {stride} frames {frames}")
            batch_size = 3
            length = frames * stride
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            for chunk_frames in [1, 2, 8]:
                if frames % chunk_frames != 0:
                    continue
                ys = []
                zs = []
                chunk_length = chunk_frames * stride
                with conv.streaming(batch_size), convtr.streaming(batch_size):
                    for offset in range(0, length, chunk_length):
                        chunk = x[..., offset : offset + chunk_length]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                y = y[..., : y_stream.shape[-1]]
                z = z[..., : z_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-6, delta
                assert frames == y_stream.shape[-1], (frames, y_stream.shape)

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()

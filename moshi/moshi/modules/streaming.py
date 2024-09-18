# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Streaming module API that should be implemented by all Streaming components,
"""

import abc
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import math
import typing as tp
from torch import nn
import torch


class Resetable(tp.Protocol):
    def reset(self) -> None:
        pass


State = tp.TypeVar("State", bound=Resetable)


class StreamingModule(abc.ABC, nn.Module, tp.Generic[State]):
    """Common API for streaming components.

    Each streaming component has a streaming state, which is just a dict[str, Tensor].
    By convention, the first dim of each tensor must be the batch size.
    Don't use dots in the key names, as this would clash with submodules
    (like in state_dict).

    If `self._is_streaming` is True, the component should use and remember
    the proper state inside `self._streaming_state`.

    To set a streaming component in streaming state, use

        with module.streaming():
            ...

    This will automatically reset the streaming state when exiting the context manager.
    This also automatically propagates to all streaming children module.

    Some module might also implement the `StreamingModule.flush` method, although
    this one is trickier, as all parents module must be StreamingModule and implement
    it as well for it to work properly. See `StreamingSequential` after.
    """

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State | None = None
        self._streaming_propagate: bool = True

    @property
    def is_streaming(self):
        return self._streaming_state is not None

    def set_streaming_propagate(self, streaming_propagate: bool):
        self._streaming_propagate = streaming_propagate

    def _apply_named_streaming(self, fn: tp.Any):
        def _handle_module(prefix: str, module: nn.Module, recurse: bool = True):
            propagate = True
            if isinstance(module, StreamingModule):
                if module._streaming_propagate:
                    fn(prefix, module)
                else:
                    propagate = False
            if not recurse:
                return
            if propagate:
                for name, child in module.named_children():
                    _handle_module(prefix + "." + name, child)

        _handle_module("", self, recurse=False)
        for name, child in self.named_children():
            _handle_module(name, child)

    def _start_streaming(self, batch_size: int):
        def _start_streaming(name: str, module: StreamingModule):
            module._streaming_state = module._init_streaming_state(batch_size)

        self._apply_named_streaming(_start_streaming)

    def _stop_streaming(self):
        def _stop_streaming(name: str, module: StreamingModule):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming)

    @abc.abstractmethod
    def _init_streaming_state(self, batch_size: int) -> State: ...

    def streaming_forever(self, batch_size: int):
        self._start_streaming(batch_size)

    @contextmanager
    def streaming(self, batch_size: int):
        """Context manager to enter streaming mode. Reset streaming state on exit."""

        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    def reset_streaming(self):
        """Reset the streaming state."""

        def _reset(name: str, module: StreamingModule):
            state = module._streaming_state
            if state is None:
                raise ValueError(
                    f"Trying to reset streaming, but {name} wasn't streaming."
                )
            state.reset()

        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> dict[str, tp.Any]:
        """Return the complete streaming state, including that of sub-modules."""
        state: dict[str, tp.Any] = {}

        def _add(name: str, module: StreamingModule):
            state[name] = module._streaming_state

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: dict[str, tp.Any]):
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name in state:
                module._streaming_state = state[name]
                state.pop(name)
            else:
                raise RuntimeError(f"Expected to find a streaming state for {name}.")

        self._apply_named_streaming(_set)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")


@dataclass
class _NullState:
    pass

    def reset(self) -> None:
        pass


class StreamingContainer(StreamingModule[_NullState]):
    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()


@dataclass
class _StreamingAddState:
    previous_x: torch.Tensor | None = None
    previous_y: torch.Tensor | None = None

    def reset(self):
        self.previous_x = None
        self.previous_y = None


class StreamingAdd(StreamingModule[_StreamingAddState]):
    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        return _StreamingAddState()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self._streaming_state is None:
            return x + y
        else:
            prev_x = self._streaming_state.previous_x
            prev_y = self._streaming_state.previous_y
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            m_l = min(x.shape[-1], y.shape[-1])
            self._streaming_state.previous_x = x[..., m_l:]
            self._streaming_state.previous_y = y[..., m_l:]
            return x[..., :m_l] + y[..., :m_l]


@dataclass
class _StreamingConvState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingConv1d(nn.Conv1d, StreamingModule[_StreamingConvState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        return _StreamingConvState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.out_channels, 0, device=input.device, dtype=input.dtype
                )
            return out


@dataclass
class _StreamingConvTrState:
    partial: torch.Tensor | None = None

    def reset(self):
        self.partial = None


class RawStreamingConvTranspose1d(
    nn.ConvTranspose1d, StreamingModule[_StreamingConvTrState]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
        return _StreamingConvTrState()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        if self._streaming_state is None:
            return super().forward(x)
        else:
            if T == 0:
                return torch.empty(
                    B, self.out_channels, 0, device=x.device, dtype=x.dtype
                )
            out = super().forward(x)
            OT = out.shape[-1]
            partial = self._streaming_state.partial
            if partial is not None:
                # Due to the potential overlap, the rightmost output of the conv transpose is not
                # ready to be output, as it will receive contributions from the next input frames.
                # Here we recover those `partial` output frames. We know that the first time step
                # of the `partial` tensor corresponds to the first time step of `out` as anything
                # coming before the first time step of `out` would have been already flushed.
                PT = partial.shape[-1]
                if self.bias is not None:
                    out[..., :PT] += partial - self.bias[:, None]
                else:
                    out[..., :PT] += partial
            # The input is T, the output is S * (T - 1) + K.
            # The offset of the left of the next frame will be S * T
            # so everything between 0 and S * T is ready to be output, and we need
            # to keep in the internal state everything beyond that, i.e. S (T - 1) + K - S T = K - S
            invalid_steps = kernel - stride
            partial = out[..., OT - invalid_steps :]
            out = out[..., : OT - invalid_steps]
            self._streaming_state.partial = partial
            return out


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
        conv = RawStreamingConv1d(chin, chout, kernel, stride).to(device)
        convtr = RawStreamingConvTranspose1d(chout, chin, kernel, stride).to(device)

        for length in [4, 8, 32, 54, 65, 128, 1043]:
            print(f"ksize {kernel} strides {stride} len {length}")
            if length < kernel:
                continue
            batch_size = 3
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            for chunk_size in [1, 3, 5, 8]:
                ys = []
                zs = []
                with conv.streaming(batch_size), convtr.streaming(batch_size):
                    for offset in range(0, length, chunk_size):
                        chunk = x[..., offset : offset + chunk_size]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                y = y[..., : y_stream.shape[-1]]
                z = z[..., : z_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-6, delta
                num_frames = int((length - kernel) / stride) + 1
                assert num_frames == y_stream.shape[-1]

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()

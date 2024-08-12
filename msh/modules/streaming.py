# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Streaming module API that should be implemented by all Streaming components,
"""

from contextlib import contextmanager
import itertools
import math
import typing as tp
from torch import nn
import torch


State = tp.Dict[str, torch.Tensor]


class StreamingModule(nn.Module):
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
        self._streaming_state: State = {}
        self._is_streaming = False

    def _apply_named_streaming(self, fn: tp.Any):
        for name, module in self.named_modules():
            if isinstance(module, StreamingModule):
                fn(name, module)

    def _set_streaming(self, streaming: bool):
        def _set_streaming(name, module):
            module._is_streaming = streaming

        self._apply_named_streaming(_set_streaming)

    @contextmanager
    def streaming(self):
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._set_streaming(True)
        try:
            yield
        finally:
            self._set_streaming(False)
            self.reset_streaming()

    def reset_streaming(self):
        """Reset the streaming state."""

        def _reset(name: str, module: StreamingModule):
            module._streaming_state.clear()

        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> State:
        """Return the streaming state, including that of sub-modules."""
        state: State = {}

        def _add(name: str, module: StreamingModule):
            if name:
                name += "."
            for key, value in module._streaming_state.items():
                state[name + key] = value

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: State):
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                # complexity is not ideal here, but probably fine.
                if key.startswith(name):
                    local_key = key[len(name) :]
                    if "." not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        assert len(state) == 0, list(state.keys())

    def flush(self, x: tp.Optional[torch.Tensor] = None):
        """Flush any remaining outputs that were waiting for completion.
        Typically, for convolutions, this will add the final padding
        and process the last buffer.

        This should take an optional argument `x`, which will be provided
        if a module before this one in the streaming pipeline has already
        spitted out a flushed out buffer.
        """
        if x is None:
            return None
        else:
            return self(x)


class StreamingAdd(StreamingModule):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self._is_streaming:
            prev_x = self._streaming_state.get("previous_x")
            prev_y = self._streaming_state.get("previous_y")
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            m_l = min(x.shape[-1], y.shape[-1])
            self._streaming_state["previous_x"] = x[..., m_l:]
            self._streaming_state["previous_y"] = y[..., m_l:]
            return x[..., :m_l] + y[..., :m_l]
        else:
            return x + y


class StreamingSequential(nn.Sequential, StreamingModule):
    """A streaming compatible alternative of `nn.Sequential`."""

    def flush(self, x: tp.Optional[torch.Tensor] = None):
        for module in self:
            if isinstance(module, StreamingModule):
                x = module.flush(x)
            elif x is not None:
                x = module(x)
        return x


class StreamingConv1d(nn.Conv1d, StreamingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

    def forward(self, x: torch.Tensor):
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        if self._is_streaming:
            # Due to the potential overlap, we potentially have some cache
            # of the previous time steps.
            previous = self._streaming_state.get("previous")
            if previous is not None:
                x = torch.cat([previous, x], dim=-1)
            B, C, T = x.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state["previous"] = x[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(x[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.out_channels, 0, device=x.device, dtype=x.dtype
                )
            return out
        else:
            return super().forward(x)

    def flush(self, x: tp.Optional[torch.Tensor] = None):
        assert self._is_streaming
        previous = self._streaming_state.pop("previous")
        if x is None:
            if previous is None:
                return None
            else:
                x = previous
        else:
            if previous is not None:
                x = torch.cat([previous, x], dim=-1)

        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        # Using ceil instead of floor here, as we will pad to the required length.
        num_frames = int(math.ceil((T - kernel) / stride) + 1)
        full_length = (num_frames - 1) * stride + kernel
        x = torch.nn.functional.pad(x, (0, full_length - T))
        return super().__call__(x)


class StreamingConvTranspose1d(nn.ConvTranspose1d, StreamingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        if self._is_streaming:
            if T == 0:
                return torch.empty(
                    B, self.out_channels, 0, device=x.device, dtype=x.dtype
                )
            out = super().forward(x)
            OT = out.shape[-1]
            partial = self._streaming_state.get("partial")
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
            self._streaming_state["partial"] = partial
            return out
        else:
            return super().forward(x)

    def flush(self, x: tp.Optional[torch.Tensor] = None):
        assert self._is_streaming
        partial = self._streaming_state.pop("partial")
        if x is None or x.shape[-1] == 0:
            if partial is None:
                return None
            return partial
        else:
            out = super().__call__(x)
            if partial is not None:
                out[..., : partial.shape[-1]] += partial
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
        conv = StreamingConv1d(chin, chout, kernel, stride).to(device)
        convtr = StreamingConvTranspose1d(chout, chin, kernel, stride).to(device)

        for length in [4, 8, 32, 54, 65, 128, 1043]:
            print(f"ksize {kernel} strides {stride} len {length}")
            if length < kernel:
                continue
            x = torch.randn(3, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            for chunk_size in [1, 3, 5, 8]:
                ys = []
                zs = []
                with conv.streaming(), convtr.streaming():
                    for offset in range(0, length, chunk_size):
                        chunk = x[..., offset : offset + chunk_size]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                    zs.append(convtr.flush())
                    y_flushed = conv.flush()
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-12, delta
                num_frames = int(math.ceil((length - kernel) / stride) + 1)
                assert num_frames == y_stream.shape[-1] + y_flushed.shape[-1]

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()

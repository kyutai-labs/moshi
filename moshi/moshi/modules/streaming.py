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
from contextlib import ExitStack
from dataclasses import dataclass
import typing as tp
from torch import nn
import torch
from ..utils.compile import CUDAGraphed


@dataclass
class State(abc.ABC):
    """Base State for streaming, requires to be resetable and also support the context
    protocol. The state __enter__ and __exit__ will be called upon entering and exiting
    streaming in the parent module, but not upon `reset()` calls.
    """

    batch_size: int
    device: torch.device

    def __post_init__(self):
        self.exec_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self._set_exec_mask_graphed: CUDAGraphed | None = None

    def set_exec_mask(self, exec_mask: torch.Tensor):
        self.exec_mask[:] = exec_mask

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.exec_mask[:] = torch.where(reset_mask, torch.ones_like(self.exec_mask), self.exec_mask)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


StateT = tp.TypeVar("StateT", bound=State)


class StreamingModule(abc.ABC, nn.Module, tp.Generic[StateT]):
    """Common API for streaming components.

    Each streaming component has a streaming state, `self._streaming_state`, which is None by default.

    To set a streaming component in streaming state, use

        with module.streaming(batch_size):
            ...

    This will automatically void the streaming state when exiting the context manager.
    This also automatically propagates to all streaming children module.
    When the streaming state is set, modules should store whatever state they need in there.
    """
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: StateT | None = None
        self._streaming_detached: bool = False
        self._cached_children: list[tuple[str, StreamingModule]] | None = None

    @property
    def is_streaming(self):
        return self._streaming_state is not None

    def set_streaming_detached(self, streaming_detached: bool):
        """If set to False, the default, this module and all submodules will switch to streaming mode
        if a parent module is set to streaming mode.
        If set to True, or in detach mode, only a direct call to this module `.streaming(...)` method
        will set it into streaming mode, ignoring the changes from its parents.

        This is useful if streaming over two different dimensions, e.g. for the RQ-Transformer
        with the inner Depth Transformer working on the dimension of the codebooks."""
        self._streaming_detached = streaming_detached

    def _apply_named_streaming(self, fn: tp.Any):
        def _handle_module(prefix: str, module: nn.Module):
            if isinstance(module, StreamingModule):
                # If prefix is empty, we are the direct receiver of the streaming request,
                # otherwise, we are inheriting from a parent and will stop if detached.
                if module._streaming_detached and prefix != "":
                    return
                assert self._cached_children is not None
                self._cached_children.append((prefix, module))
            for name, child in module.named_children():
                if prefix:
                    new_prefix = prefix + "." + name
                else:
                    new_prefix = name
                _handle_module(new_prefix, child)

        if self._cached_children is None:
            self._cached_children = []
            _handle_module("", self)
        for name, child in self._cached_children:
            fn(name, child)

    def _start_streaming(self, batch_size: int, exit_stack: ExitStack):
        def _start_streaming(name: str, module: StreamingModule):
            assert module._streaming_state is None, f"{name} is already streaming!"
            state = module._init_streaming_state(batch_size)
            exit_stack.enter_context(state)
            module._streaming_state = state

        self._apply_named_streaming(_start_streaming)

    def _stop_streaming(self) -> None:
        def _stop_streaming(name: str, module: StreamingModule):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming)

    @abc.abstractmethod
    def _init_streaming_state(self, batch_size: int) -> StateT: ...

    def streaming_forever(self, batch_size: int):
        self.streaming(batch_size).__enter__()

    def streaming(self, batch_size: int) -> ExitStack:
        """Context manager to enter streaming mode. Reset streaming state on exit."""

        exit_stack = ExitStack()
        self._start_streaming(batch_size, exit_stack)
        exit_stack.callback(self._stop_streaming)
        return exit_stack

    def reset_streaming(self, reset_mask: torch.Tensor | None = None) -> None:
        """Reset the streaming state."""

        def _reset(name: str, module: StreamingModule):
            state = module._streaming_state
            if state is None:
                raise ValueError(
                    f"Trying to reset streaming, but {name} wasn't streaming."
                )
            state.reset(reset_mask)

        state = self._streaming_state
        assert state is not None
        if reset_mask is None:
            reset_mask = torch.ones(state.batch_size, device=state.device, dtype=torch.bool)
        else:
            reset_mask = reset_mask.to(state.device)
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

    def set_exec_mask(self, exec_mask: torch.Tensor):
        """Set the execution mask, a tensor of boolean of shape `(B,), indicating
        for each batch item whether the internal state should be updated or not as if
        real data had been received.

        This is useful for running desynchronized streams with batching, e.g. when
        the mask is False for an entry, the internal state will be unchanged by the provided
        data, e.g. will be on the next step as if the previous one had never happened.

        There is no magic here, each StreamingModule subclass is responsible for respecting
        the exec_mask.
        """

        state = self._streaming_state
        assert state is not None
        exec_mask = exec_mask.to(state.device)

        def _set_exec_mask(exec_mask: torch.Tensor):
            def _set_exec_mask_fn(name: str, module: StreamingModule):
                state = module._streaming_state
                assert state is not None
                state.set_exec_mask(exec_mask)
            self._apply_named_streaming(_set_exec_mask_fn)

        if state._set_exec_mask_graphed is None:
            disable = state.device.type != 'cuda'
            state._set_exec_mask_graphed = CUDAGraphed(_set_exec_mask, disable=disable)

        state._set_exec_mask_graphed(exec_mask)


class StreamingContainer(StreamingModule[State]):
    def _init_streaming_state(self, batch_size: int) -> State:
        device = next(iter(self.parameters())).device
        return State(batch_size, device)

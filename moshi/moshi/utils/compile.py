# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides some extra utilities around torch compile, in particular with a way
to fully deactivate it easily with a context manager.
Provides a simple activation checkpointing that is compatible with FSDP and torch compile.
Finally, provides some utilities for CUDA graphing functions.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import os
import typing as tp

import torch
from torch import cuda


_compile_disabled: bool = False


@contextmanager
def no_compile():
    """Disable torch.compile locally. Now Pytorch 2.4 provides a function to do that."""
    global _compile_disabled

    prev_disabled = _compile_disabled
    _compile_disabled = True
    try:
        yield
    finally:
        _compile_disabled = prev_disabled


def torch_compile_lazy(fun):
    """torch.compile creates a huge pool of processes, even when not using the function at all,
    e.g. with Dora. This can polute stderr when doing CTRL+C. So we do it in a lazy way.
    """
    if os.environ.get("NO_TORCH_COMPILE"):
        return fun
    fun_compiled = None

    @wraps(fun)
    def _wrapped(*args, **kwargs):
        nonlocal fun_compiled
        if _compile_disabled:
            return fun(*args, **kwargs)
        if fun_compiled is None:
            fun_compiled = torch.compile(fun)
        return fun_compiled(*args, **kwargs)

    return _wrapped


class Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, *args) -> tp.Any:
        to_save = []
        ctx.others = []
        ctx.function = function
        # Sources will indicate whether the arg in position N is
        # a tensor stored in ctx.save_for_backward, or inside ctx.others.
        ctx.sources = []
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                to_save.append(arg)
                ctx.sources.append("tensor")
                new_args.append(arg.detach())
            else:
                ctx.sources.append("other")
                ctx.others.append(arg)
                new_args.append(arg)
        ctx.save_for_backward(*to_save)
        # During the forward, we just make a pass with no gradient computed.
        with torch.no_grad():
            res = function(*new_args)
        return res

    @staticmethod
    def backward(ctx, *grads) -> tp.Tuple[tp.Optional[torch.Tensor], ...]:
        pseudo_tensors = []
        with torch.set_grad_enabled(True):
            # We create leaf tensors to collect the output gradients.
            # We call them pseudo_tensors because they are pretending to be the input
            # to `function` but are not directly
            for tensor in ctx.saved_tensors:
                pseudo_tensor = tensor.detach()
                pseudo_tensor.requires_grad_(True)
                pseudo_tensors.append(pseudo_tensor)
            pseudo_tensors_copy = list(pseudo_tensors)
            args = []
            for source in ctx.sources:
                if source == "other":
                    args.append(ctx.others.pop(0))
                else:
                    assert source == "tensor"
                    args.append(pseudo_tensors_copy.pop(0))
            res = ctx.function(*args)
            # The second forward with grad computation allows us to connect the input leaf tensors
            # inside pseudo_tensors, to the outputs of the function called.
        if not isinstance(res, tuple):
            res = (res,)
        # Now we just ask Torch to compute the derivative of `res` given the gradient coming from above
        # `grads`. The computed gradient will end up into the `pseudo_tensors` grad attributes.
        torch.autograd.backward(res, grads)
        out: tp.List[tp.Optional[torch.Tensor]] = [None]
        for source in ctx.sources:
            # We still need to output `None` values for non tensor parameters.
            if source == "other":
                out.append(None)
            else:
                assert source == "tensor"
                out.append(pseudo_tensors.pop(0).grad)
        return tuple(out)


def simple_checkpoint(module: torch.nn.Module, *args, **kwargs):
    """Custom implementation of checkpointing in PyTorch as the builtin implementation is broken
    when using torch compile. Only supports wrapping a `nn.Module` with a forward with no `*args` or `**kwargs`.

    https://github.com/pytorch/pytorch/issues/97436.
    Should be resolved in nightlies, but it is quite fun and simple to code it ourselves.
    """
    if hasattr(module, "_fsdp_wrapped_module"):
        module_for_sig = module._fsdp_wrapped_module
    else:
        module_for_sig = module
    assert isinstance(module_for_sig, torch.nn.Module)
    sig = inspect.signature(module_for_sig.forward)
    # We first flatten all arguments to use only *args, to make things easier and because
    # torch.autograd.Function has weird support for kwargs.
    bounded = sig.bind(*args, **kwargs)
    new_args = []
    for name, param in sig.parameters.items():
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise RuntimeError("simple_checkpoint doesn't support var args.")
        if name not in bounded.arguments:
            break
        new_args.append(bounded.arguments[name])
    return Checkpoint.apply(module, *new_args)


_in_cuda_graph = False
_disable_cuda_graph = False


def in_cuda_graph() -> bool:
    """Indicate whether we are in a function that is CUDA Graphed (or will be soon)."""
    return _in_cuda_graph


@contextmanager
def _set_in_cuda_graph():
    global _in_cuda_graph
    assert not _in_cuda_graph
    _in_cuda_graph = True
    try:
        yield
    finally:
        _in_cuda_graph = False


def _is_cuda_graph_enabled() -> bool:
    if _disable_cuda_graph:
        return False
    no_cuda_graph = os.environ.get("NO_CUDA_GRAPH", "")
    if no_cuda_graph.lower() not in {"0", "no", "n", ""}:
        return False
    return True


@contextmanager
def no_cuda_graph():
    """Deactivate CUDA Graphing for all the calls in this context manager."""
    global _disable_cuda_graph
    old_value = _disable_cuda_graph
    _disable_cuda_graph = True
    try:
        yield
    finally:
        _disable_cuda_graph = old_value


class CUDAGraphed:
    """Allow simple CUDA Graphing of a function.

    Args:
        func: callable, taking any number of arguments. Its tensors arguments should
            be top level args, not nested in structures (tuples, dicts, etc). Keyword
            arguments are NOT supported for simplicity.
        warmup_steps: how many call to make normally before CUDA Graphing. In particular, this
            allows torch.compiled functions to get properly compiled.
        disabled: if True, just call the func directly, useful to quickly deactivate on CPU.
    """

    def __init__(self, func: tp.Callable, warmup_steps: int = 1, disable: bool = False):
        self.func = func
        self.warmup_steps = warmup_steps
        self.disable = disable
        self._graph: cuda.CUDAGraph | None = None
        self._output: tuple | None = None
        self._args: tuple | None = None

    def reset(self, warmup_steps: int = 0) -> None:
        """Reset the state, meaning the next call we get CUDA Graphed again. Useful if some
        shapes have changed, or external state (e.g. KVCache) has changed."""
        self.warmup_steps = warmup_steps
        self._graph = None
        self._output = None
        self._args = None

    def __call__(self, *args, **kwargs) -> tp.Any:
        if kwargs:
            raise RuntimeError("Named arguments not supported for now.")
        if self.disable or not _is_cuda_graph_enabled() or in_cuda_graph():
            return self.func(*args, **kwargs)

        def _clone_tensors(args: tuple) -> tuple:
            out: list = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = arg.clone()
                out.append(arg)
            return tuple(out)

        def _match_values_copy_tensors(args: tuple, target_args: tuple) -> None:
            if len(args) != len(target_args):
                raise ValueError(
                    f"Expected {len(target_args)}, but got {args} for CUDA Graphed function."
                )
            for idx, (source, target) in enumerate(zip(args, target_args)):
                if isinstance(target, torch.Tensor):
                    if not isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"Argument #{idx} was a tensor, and is no longer (now {source})."
                        )
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Argument #{idx} had shape {target.shape}, but got shape {source.shape}"
                            "When using CUDAGraph, every call must be done with exactly the same shapes. "
                            "Feel free to deactivate with the env variable NO_CUDA_GRAPH=1, or the decorator "
                            "`with no_cuda_graph():`"
                        )
                    target.copy_(source)
                else:
                    if isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"Argument #{idx} was not a tensor {target}, but is now one."
                        )
                    if source is not target and source != target:
                        raise ValueError(
                            f"Argument #{idx} changed value from {target} to {source}."
                        )

        with _set_in_cuda_graph():
            # Prevent any one under us to try and CUDA Graph things.
            if self._graph is None:
                if self.warmup_steps <= 0:
                    self._graph = cuda.CUDAGraph()
                    # Making a copy just to ensure those are not used else where.
                    self._args = _clone_tensors(args)
                    with cuda.graph(self._graph):
                        self._output = self.func(*self._args)
                    # At this point nothing really happened, so we have to make it run for real.
                    self._graph.replay()
                    return self._output
                else:
                    self.warmup_steps -= 1
                    return self.func(*args)
            else:
                assert self._args is not None
                _match_values_copy_tensors(args, self._args)
                self._graph.replay()
                return self._output


def cuda_graph(func: tp.Callable, warmup_steps: int = 1):
    """Just calls `CUDAGraphed` on the given function."""
    if not _is_cuda_graph_enabled():
        return func
    return CUDAGraphed(func, warmup_steps)

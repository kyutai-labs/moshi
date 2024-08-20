# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""torch compile utilities."""
from contextlib import contextmanager
from functools import wraps
import inspect
import os
import typing as tp

import torch


_compile_disabled: bool = False


@contextmanager
def no_compile():
    """Disable torch.compile locally."""
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
    if os.environ.get('NO_TORCH_COMPILE'):
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
                ctx.sources.append('tensor')
                new_args.append(arg.detach())
            else:
                ctx.sources.append('other')
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
                if source == 'other':
                    args.append(ctx.others.pop(0))
                else:
                    assert source == 'tensor'
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
            if source == 'other':
                out.append(None)
            else:
                assert source == 'tensor'
                out.append(pseudo_tensors.pop(0).grad)
        return tuple(out)


def simple_checkpoint(module: torch.nn.Module, *args, **kwargs):
    """Custom implementation of checkpointing in PyTorch as the builtin implementation is broken
    when using torch compile. Only supports wrapping a `nn.Module` with a forward with no `*args` or `**kwargs`.

    https://github.com/pytorch/pytorch/issues/97436.
    Should be resolved in nightlies, but it is quite fun and simple to code it ourselves.
    """
    if hasattr(module, '_fsdp_wrapped_module'):
        module_for_sig = module._fsdp_wrapped_module
    else:
        module_for_sig = module
    sig = inspect.signature(module_for_sig.forward)
    # We first flatten all arguments to use only *args, to make things easier and because
    # torch.autograd.Function has weird support for kwargs.
    bounded = sig.bind(*args, **kwargs)
    new_args = []
    for name, param in sig.parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise RuntimeError("simple_checkpoint doesn't support var args.")
        if name not in bounded.arguments:
            break
        new_args.append(bounded.arguments[name])
    return Checkpoint.apply(module, *new_args)

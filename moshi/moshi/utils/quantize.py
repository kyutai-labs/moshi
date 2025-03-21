# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization based on bitsandbytes, supporting only 8 bits for now.
We are taking from freedom from the intended use of bnb:

- we are not replacing Linear with Linear8bitLt, but rely instead of the explicit use
    of the `linear(module, x)` function.
- for multi linears (e.g. per timestep weights in the Depth Transformer), we instead use the
    `multi_linear` function.
"""

import torch
from torch import nn


def linear(module: nn.Module, x: torch.Tensor, name='weight') -> torch.Tensor:
    import bitsandbytes as bnb  # type: ignore
    if is_quantized(module, name):
        state = bnb.MatmulLtState()
        state.CB = getattr(module, name)
        assert isinstance(state.CB, torch.Tensor)
        state.SCB = getattr(module, name + '_scb')
        assert isinstance(state.SCB, torch.Tensor)
        assert state.SCB.dtype == torch.float, state.SCB.dtype
        state.has_fp16_weights = False
        y = bnb.matmul(x.half(), state.CB, state=state)
        assert isinstance(y, torch.Tensor)
        return y
    else:
        return module(x)


def multi_linear(num_steps: int, schedule: list[int] | None,
                 module: nn.ModuleList, x: torch.Tensor, offset: int, name='weight') -> torch.Tensor:
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        num_steps (int): Number of possible time steps.
        schedule (list[int] or None): schedule for weight sharing.
        weight (torch.Tensor): Weight tensor, with shape `[num_linear * chout, chin]`.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """
    import bitsandbytes as bnb  # type: ignore
    B, T, C = x.shape
    ys: list[torch.Tensor] = []
    if are_quantized(module, name):
        #TODO test Multi linear with quantized weights
        weights = [getattr(mod, name) for mod in module]
        weights_scb = [getattr(mod, name + '_scb') for mod in module]
    else:
        weights = None
        weights_scb = None

    if weights_scb is not None:
        _, chout = weights[0].shape
        assert all([weight_scb.shape == (chout,) for weight_scb in weights_scb])
        assert all([weight_scb.dtype == torch.float for weight_scb in weights_scb])

    for t in range(T):
        linear_index = t + offset
        if schedule is not None:
            linear_index = schedule[linear_index]
        if weights_scb is None:
            y = module[linear_index](x[:, t])
        else:
            state = bnb.MatmulLtState()
            CB = weights[linear_index]
            state.CB = CB  # type: ignore
            state.SCB = weights_scb[linear_index]
            state.has_fp16_weights = False
            y = bnb.matmul(x[:, t].half(), CB, state=state)
            assert isinstance(y, torch.Tensor)
        ys.append(y)
    out = torch.stack(ys, 1)
    return out


def is_quantized(module: nn.Module, name: str = 'weight'):
    return hasattr(module, name + '_scb')

def are_quantized(module: nn.ModuleList, name: str | list[str] = 'weight') -> bool:
    return all(is_quantized(mod, name) for mod in module)


def quantize_param(module: nn.Module, name: str = 'weight') -> None:
    from bitsandbytes import functional as bnbF  # type: ignore
    if is_quantized(module, name):
        # Due to model casting, the type of SCB might be wrong, althought
        # that would only happen during the init. Let's recast it to float.
        SCB = getattr(module, name + '_scb')
        if SCB.dtype != torch.float:
            setattr(module, name + '_scb', nn.Parameter(SCB.to(torch.float), requires_grad=False))
        return
    weight = getattr(module, name)
    assert weight.data.dtype.is_floating_point
    CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))  # type: ignore
    setattr(module, name, nn.Parameter(CB, requires_grad=False))
    setattr(module, name + '_scb', nn.Parameter(SCB, requires_grad=False))


def quantize_linear(linear: nn.Module) -> None:
    assert linear.bias is None
    quantize_param(linear)

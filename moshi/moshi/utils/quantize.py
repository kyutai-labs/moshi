# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization based on bitsandbytes, supporting only 8 bits for now.
We are taking from freedom from the intended use of bnb:
"""

import torch
from torch import nn


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        from bitsandbytes import functional as bnbF  # type: ignore
        weight = linear.weight
        assert weight.data.dtype.is_floating_point
        assert linear.bias is None
        CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))  # type: ignore
        self.weight = nn.Parameter(CB, requires_grad=False)
        self.weight_scb = nn.Parameter(SCB, requires_grad=False)

    def forward(self, x):
        import bitsandbytes as bnb  # type: ignore
        state = bnb.MatmulLtState()
        state.CB = self.weight  # type: ignore
        assert isinstance(state.CB, torch.Tensor)
        state.SCB = self.weight_scb  # type: ignore
        assert isinstance(state.SCB, torch.Tensor)
        if state.SCB.dtype != torch.float:
            raise RuntimeError(
                "Expected `weight_scb` to have type float, but got bfloat16. "
                "When using quantized models, care should be taken not to change the dtype of "
                "the model once initialized.")
        assert state.SCB.dtype == torch.float, state.SCB.dtype
        state.has_fp16_weights = False
        y = bnb.matmul(x.half(), state.CB, state=state)
        assert isinstance(y, torch.Tensor)
        return y


def replace_linear_with_qlinear(module):
    """Recursively replace all Linear layers with QLinear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, QLinear(child))
        elif isinstance(child, QLinear):
            # Slight issue with the way we implement things: the scale param
            # might get casted with the rest of the model to bfloat16, altough
            # we most likely want to keep it as float. For the LM model we might call this function twice,
            # first layer by layer to avoid to big of a memory usage, and second, at the end
            # of the LM init, after all other modules are initialized and properly dtyped.
            # In any case that should happen before loading the state dict to avoid a loss of precision.
            child.float()
        else:
            replace_linear_with_qlinear(child)

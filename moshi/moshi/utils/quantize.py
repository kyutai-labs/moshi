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


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        from bitsandbytes import functional as bnbF  # type: ignore
        weight = linear.weight
        assert weight.data.dtype.is_floating_point
        assert linear.bias is None
        CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))  # type: ignore
        self.weight = nn.Parameter(CB, requires_grad=False)
        self.weight_scb = nn.Parameter(SCB, requires_grad=False)

    def foward(self, x):
        import bitsandbytes as bnb  # type: ignore
        state = bnb.MatmulLtState()
        state.CB = self.weight
        assert isinstance(state.CB, torch.Tensor)
        state.SCB = self.weight_scb
        assert isinstance(state.SCB, torch.Tensor)
        assert state.SCB.dtype == torch.float, state.SCB.dtype
        state.has_fp16_weights = False
        y = bnb.matmul(x.half(), state.CB, state=state)
        assert isinstance(y, torch.Tensor)
        return y


def quantize_linear(linear: nn.Module) -> None:
    assert linear.bias is None
    quantize_param(linear)

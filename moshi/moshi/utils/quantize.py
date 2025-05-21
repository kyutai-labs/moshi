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
        super().__init__()
        from bitsandbytes import functional as bnbF  # type: ignore
        weight = linear.weight
        assert weight.data.dtype.is_floating_point
        assert linear.bias is None

        # Check if the weight is on a meta device
        if weight.device.type == 'meta':
            # For meta device, we need to preserve the shape information
            # Create tensors with the same shape as the original weight
            # We'll use the shape information from the linear layer
            out_features, in_features = weight.shape

            # Create CB tensor with shape [out_features * 8/8, in_features]
            # The first dimension is rounded up to a multiple of 8
            # This matches the shape that would be produced by int8_vectorwise_quant
            padded_out_features = ((out_features + 7) // 8) * 8
            self.weight = nn.Parameter(
                torch.zeros((padded_out_features, in_features),
                           dtype=torch.int8, device='meta'),
                requires_grad=False
            )

            # Create SCB tensor with shape [out_features]
            self.weight_scb = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float, device='meta'),
                requires_grad=False
            )
            self.is_meta = True
        else:
            # Normal quantization for non-meta tensors
            CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))  # type: ignore
            self.weight = nn.Parameter(CB, requires_grad=False)
            self.weight_scb = nn.Parameter(SCB, requires_grad=False)
            self.is_meta = False

    def _check_meta_status(self):
        """Check if the weights are still meta tensors and update is_meta flag accordingly."""
        if hasattr(self, 'is_meta') and self.is_meta:
            # Check if the weights have been loaded (no longer on meta device)
            if self.weight.device.type != 'meta' and self.weight.numel() > 0:
                self.is_meta = False

                # Ensure the scale tensor is float32, regardless of the model's dtype
                if self.weight_scb.dtype != torch.float:
                    self.weight_scb.data = self.weight_scb.data.float()

    def forward(self, x):
        import bitsandbytes as bnb  # type: ignore

        # Update meta status based on actual tensor properties
        self._check_meta_status()

        # Check if this is a meta tensor that hasn't been properly initialized yet
        if hasattr(self, 'is_meta') and self.is_meta:
            # If we're still in meta mode but trying to do a forward pass,
            # this means the weights weren't properly loaded
            raise RuntimeError(
                "Attempting to run forward pass with meta tensors. "
                "The model weights need to be loaded before running inference.")

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

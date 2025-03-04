# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.compile import torch_compile_lazy
from ..utils import quantize
from ..modules.lora import LoraArgs, maybe_lora_layer

@torch_compile_lazy
def gating_forward_kernel(
    weight_in: torch.Tensor, weight_out: torch.Tensor, activation, x: torch.Tensor
):
    x = F.linear(x, weight_in)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = F.linear(x, weight_out)
    return x


# If torch_compile_lazy => AssertionError: Dynamo only supports FSDP with use_orig_params=True
def gating_forward_lora_kernel(
    linear_in: nn.Module, linear_out: nn.Module, activation, x: torch.Tensor
):
    x = linear_in(x)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = linear_out(x)
    return x

class ActivationGating(nn.Module):
    """
    Gating FFN layer, using the given activation.
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
        **factory_kwargs: other kwargs passed to the linear layer, in particular device and dtype.
    """

    _fsdp_final = True

    def __init__(self, dim: int, dim_feedforward: int, activation, quantized: bool = False, lora_args: LoraArgs | None = None ,**factory_kwargs):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3

        MaybeLora = maybe_lora_layer(lora_args)
        self.lora =  (not lora_args is None and lora_args.enable)
        self.linear_in = MaybeLora(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = MaybeLora(
                hidden, dim, bias=False, **factory_kwargs
            )
        # We try to follow the default PyTorch MHA convention, to easily compare results.

        self.activation = activation

    def forward(self, x: torch.Tensor):
        if quantize.is_quantized(self.linear_in):
            assert quantize.is_quantized(self.linear_out)
            x = quantize.linear(self.linear_in, x)
            B, T, _ = x.shape
            x = x.view(B, T, 2, -1)
            x = self.activation(x[..., 0, :]) * x[..., 1, :]
            x = quantize.linear(self.linear_out, x)
            return x

        if not self.lora:
            return gating_forward_kernel(
                self.linear_in.weight, self.linear_out.weight, self.activation, x
            )
        else:
            return gating_forward_lora_kernel(
                self.linear_in, self.linear_out, self.activation, x
            )


def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(torch.nn.functional, name)
    elif name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def _make_gating(
    name: str, dim: int, dim_feedforward: int, lora_args: LoraArgs | None,
    **factory_kwargs
) -> nn.Module:
    return ActivationGating(
        dim, dim_feedforward, _get_activation(name), lora_args = lora_args, **factory_kwargs
    )


def make_gating(
    name: str, dim: int, dim_feedforward: int, lora_args: LoraArgs | None, **factory_kwargs
) -> nn.Module:
    gating = _make_gating(name, dim, dim_feedforward, lora_args = lora_args, **factory_kwargs)
    max_params = 2 * dim * dim_feedforward
    params = sum(p.numel() for p in gating.parameters())
    if lora_args is None or not lora_args.enable:
        assert (
            params <= max_params
        ), f"{name} gating has {params} params, max is {max_params}"
    return gating

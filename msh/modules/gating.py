# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..utils.utils import torch_compile_lazy


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


class ActivationGating(nn.Module):
    """
    Gating FFN layer, using the given activation.
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
        **factory_kwargs: other kwargs passed to the linear layer, in particular device and dtype.
    """

    _fsdp_final = True

    def __init__(self, dim: int, dim_feedforward: int, activation, **factory_kwargs):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        return gating_forward_kernel(
            self.linear_in.weight, self.linear_out.weight, self.activation, x
        )


@torch_compile_lazy
def _apply_svd_rot(phase1, phase2, svd_values, z):
    z1 = z[:, :, :, 0]
    z2 = z[:, :, :, 1]

    svd1 = svd_values[:, :, :, 0]
    svd2 = svd_values[:, :, :, 1]

    zi1 = svd1 * (torch.cos(phase1) * z1 - torch.sin(phase1) * z2)
    zi2 = svd2 * (torch.sin(phase1) * z1 + torch.cos(phase1) * z2)

    zo1 = torch.cos(phase2) * zi1 - torch.sin(phase2) * zi2
    zo2 = torch.sin(phase2) * zi1 + torch.cos(phase2) * zi2

    return torch.stack([zo1, zo2], dim=-1)


class SVDGating(nn.Module):
    """
    Experimental module for performing arbitrary 2x2 gating using a SVD based parametrization with
    1 diagonal matrix and 2 rotation matrices.
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
        small_init (bool): when True, initializes the rotations at 0, and the eigen values closer to 0.
        tanh_rot (bool): when True, the angles for the rotations are passed through a tanh.
        tanh_scale (bool): the range of the tanh for the rotations will be `-tanh_scale * pi` to `+tanh_scale * pi`.
        **factory_kwargs: other kwargs passed to the linear layer, in particular device and dtype.
    """

    _fsdp_final = True

    def __init__(
        self,
        dim: int,
        activation,
        small_init: bool = True,
        tanh_rot: bool = False,
        tanh_scale: float = 1,
        scale_rot: float = 0.1,
        **factory_kwargs,
    ):
        super().__init__()
        # what should be effective dim to match the param of simple linear ?
        # normally we have d -> 4 d -> d
        # e.g. 2 * d * 4 d = 8 d^2 params
        # Now let's note `h` hidden dimension we will use.
        # we have 2 * d * h + d * 2 * h = 4 d h
        # so h = 2 * d
        h = 2 * dim
        self.dim = dim
        self.activation = activation
        self.scale_rot = scale_rot
        self.tanh_rot = tanh_rot
        self.tanh_scale = tanh_scale
        self.w_in = nn.Linear(dim, h, bias=False, **factory_kwargs)
        self.w_gate = nn.Linear(dim, 2 * h, bias=False, **factory_kwargs)
        self.w_out = nn.Linear(h, dim, bias=False, **factory_kwargs)
        if small_init:
            self.w_gate.weight.data.view(-1, 4, dim)[:, 2:].zero_()
            self.w_gate.weight.data.view(-1, 4, dim)[:, :2] *= 0.1

    def forward(self, x):
        z = self.w_in(x)
        mat = self.w_gate(x)
        B, T, H = z.shape
        z = z.view(B, T, H // 2, 2)

        mat = mat.view(B, T, H // 2, 4)
        svd_values = self.activation(mat[..., :2])
        if self.tanh_rot:
            rot_phase1 = (self.tanh_scale * math.pi) * torch.tanh(mat[..., 2])
            rot_phase2 = (self.tanh_scale * math.pi) * torch.tanh(mat[..., 3])
        else:
            rot_phase1 = self.scale_rot * 2 * math.pi * mat[..., 2]
            rot_phase2 = self.scale_rot * 2 * math.pi * mat[..., 3]
        z = _apply_svd_rot(rot_phase1, rot_phase2, svd_values, z)

        z = z.view(B, T, H)
        return self.w_out(z)


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
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    if name.startswith("svd_"):
        activation = name.split("_", 1)[1]
        kwargs = {}
        if ":" in activation:
            activation, params = activation.split(":", 1)
            key_values = [p.strip().split("=", 1) for p in params.split(",")]
            for key, value in key_values:
                kwargs[key] = eval(value)
        assert dim_feedforward == dim * 4
        return SVDGating(dim, _get_activation(activation), **kwargs, **factory_kwargs)
    elif name == "xformers_swiglu":
        # For compatiblity with previous runs, now everything goes through the torch compiled version.
        return ActivationGating(
            dim, dim_feedforward, _get_activation("silu"), **factory_kwargs
        )
    else:
        return ActivationGating(
            dim, dim_feedforward, _get_activation(name), **factory_kwargs
        )


def make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    gating = _make_gating(name, dim, dim_feedforward, **factory_kwargs)
    max_params = 2 * dim * dim_feedforward
    params = sum(p.numel() for p in gating.parameters())
    assert (
        params <= max_params
    ), f"{name} gating has {params} params, max is {max_params}"
    return gating

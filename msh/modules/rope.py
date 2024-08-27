# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import math
import torch
from ..utils.utils import torch_compile_lazy


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor, k: torch.Tensor, offset: int = 0, max_period: float = 10_000
):
    """
    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offset (int): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
    """

    B, T, H, D = q.shape
    assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = torch.arange(offset, offset + T, device=q.device, dtype=torch.float32).view(
        -1, 1, 1
    )

    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, H, D // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(B, T, H, D), ko.view(B, T, H, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offset, self.max_period)

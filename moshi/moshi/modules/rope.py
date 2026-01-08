# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import math
import torch
from ..utils.compile import torch_compile_lazy


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    interleave: bool = True,
    time_before_heads: bool = False,
):
    """
    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offset (int): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
        interleave (bool): If True, real and imaginarie part are interleaved
        time_before_heads (bool):  if True, expected [B, T, H, D], else [B, H, T ,D]
    """

    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape

    assert q.shape[0] == k.shape[0]   # B
    if time_before_heads:
        assert q.shape[1] == k.shape[1]  # T
        assert q.shape[3] == k.shape[3]  # D
    else:
        assert q.shape[2] == k.shape[2]  # T
        assert q.shape[3] == k.shape[3]  # D

    assert D % 2 == 0

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)

    if time_before_heads:
        ts = ts.view(B, -1, 1, 1)
    else:
        ts = ts.view(B, 1, -1, 1)

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    if interleave:
        # [r0,i0,r1,i1,...]
        q = q.view(*q.shape[:-1], D // 2, 2)
        k = k.view(*k.shape[:-1], D // 2, 2)
        qr, qi = q[..., 0].float(), q[..., 1].float()
        kr, ki = k[..., 0].float(), k[..., 1].float()
    else:
        # [r..., i...]
        qr, qi = q[..., : D // 2].float(), q[..., D // 2 :].float()
        kr, ki = k[..., : D // 2].float(), k[..., D // 2 :].float()

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    if interleave:
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1).view(*q.shape[:-2], D)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1).view(*k.shape[:-2], D)
    else:
        qo = torch.cat([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.cat([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo, ko


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, interleave: bool, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period
        self.interleave = interleave

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offset, self.max_period, self.interleave, time_before_heads)

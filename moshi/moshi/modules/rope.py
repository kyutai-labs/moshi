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
    time_before_heads: bool = False,
    positions: torch.Tensor | None = None,
):
    """
    Apply rotary embeddings to q and k.

    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]` if `time_before_heads` else `[B, H, T, D]`.
        k (torch.Tensor): keys, same shape as q.
        offset (torch.Tensor): streaming offset of shape [B] or scalar.
        max_period (float): maximum period for the cos and sin rotations.
        time_before_heads (bool): format flag for input shapes.
        positions (torch.Tensor | None): optional tensor of positions of shape [B, T].
                                         Used instead of offset + arange(T).
    """
    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape

    assert k.shape == q.shape
    assert D % 2 == 0
    assert max_period > 0

    # Compute frequencies: shape [D/2]
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))  # [D/2]

    # Compute positions: shape [B, T]
    if positions is not None:
        ts = positions.float()
        valid_mask = ts != -1

        ts = torch.where(valid_mask, ts, 0.)
    else:
        base = offset.float().view(-1, 1)  # [B, 1]
        ts = base + torch.arange(T, device=q.device, dtype=torch.float32)  # [B, T]
        valid_mask = None
    # Reshape ts to match shape for broadcasting with q and k
    if time_before_heads:
        ts = ts.view(B, T, 1, 1)  # for [B, T, H, D]
    else:
        ts = ts.view(B, 1, T, 1)  # for [B, H, T, D]

    # Expand freqs for broadcasting: [D/2] -> [1, 1, 1, D/2]
    freqs = freqs.view(*([1] * (ts.ndim - 1)), -1)

    # Reshape q/k to split real and imaginary parts
    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)

    qr, qi = q[..., 0], q[..., 1]
    kr, ki = k[..., 0], k[..., 1]

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    qo = qo.view(*dims, D)
    ko = ko.view(*dims, D)

    if valid_mask is not None:
        valid_mask = valid_mask.view(B, 1, T, 1).expand(B, 1, T, D)

        qo = torch.where(valid_mask, qo, q.view(*dims, D))
        ko = torch.where(valid_mask, ko, k.view(*dims, D))
    return qo, ko


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
        positions: torch.Tensor | None = None
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offset, self.max_period, time_before_heads, positions)

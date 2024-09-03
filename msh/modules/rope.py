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

    def __init__(
        self, dim: int, max_len: int, device=None, max_period: float = 10000.0
    ):
        super().__init__()

        assert dim > 0
        assert dim % 2 == 0
        assert max_period > 0

        self.max_period = max_period
        self.max_len = max_len
        self.dim = dim

        ds = torch.arange(dim // 2, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2 / dim))
        ts = torch.arange(max_len, device=device, dtype=torch.float32).view(-1, 1, 1)
        self.rotr = torch.cos(freqs * ts)
        self.roti = torch.sin(freqs * ts)
        self.cuda_graph = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rope rotation to query or key tensor."""
        B, T, H, D = q.shape
        if T == 1:
            if self.cuda_graph is None:
                self.cuda_graph = torch.cuda.CUDAGraph()
                self.in_q = q.clone()
                self.in_k = k.clone()
                self.in_rotr = self.rotr[offset : offset + T].clone()
                self.in_roti = self.roti[offset : offset + T].clone()
                self.current_offset = offset
                with torch.cuda.graph(self.cuda_graph):
                    q = self.in_q.view(B, T, H, D // 2, 2)
                    k = self.in_k.view(B, T, H, D // 2, 2)

                    # convention is `r` suffix is real part, `i` is imaginary.
                    qr = q[..., 0].float()
                    qi = q[..., 1].float()

                    kr = k[..., 0].float()
                    ki = k[..., 1].float()

                    qor = qr * self.in_rotr - qi * self.in_roti
                    qoi = qr * self.in_roti + qi * self.in_rotr

                    kor = kr * self.in_rotr - ki * self.in_roti
                    koi = kr * self.in_roti + ki * self.in_rotr

                    dtype = q.dtype
                    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
                    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

                    self.out_q = qo.view(B, T, H, D)
                    self.out_k = ko.view(B, T, H, D)
            else:
                self.in_q.copy_(q)
                self.in_k.copy_(k)
                if self.current_offset != offset:
                    self.in_rotr.copy_(self.rotr[offset : offset + T])
                    self.in_roti.copy_(self.roti[offset : offset + T])
                    self.current_offset = offset
            self.cuda_graph.replay()
            return (self.out_q, self.out_k)

        return apply_rope(q, k, offset, self.max_period)

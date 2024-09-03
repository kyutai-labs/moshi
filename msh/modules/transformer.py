# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, xformer attention support
and easy causal attention with a potentially finite receptive field.

See `StreamingTransformer` for more information.

Unlike regular PyTorch Transformer, we make the hard choice that batches are first.
"""

import math
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from xformers import ops

from .gating import make_gating
from .rope import RotaryEmbedding
from .streaming import StreamingModule
from ..utils.compile import torch_compile_lazy

from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    _SeqLenInfo,
    _PaddedSeqLenInfo,
)


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: tp.Optional[torch.dtype],
    eps: float,
):
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm", "real_rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32", "real_rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def multi_linear(
    num_linear: int, weight: torch.Tensor, x: torch.Tensor, offset: int,
):
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        num_linear (int): Number of possible time steps and so number of linears.
        weight (torch.Tensor): Weight tensor, with shape `[num_linear * chout, chin]`.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """
    B, T, C = x.shape
    ys = []
    chout, chin = weight.shape
    weight = weight.view(num_linear, -1, chin)
    for t in range(T):
        y = F.linear(x[:, t], weight[t + offset])
        ys.append(y)
    out = torch.stack(ys, 1)
    return out


def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    """Deactivates or changes the context span (in time steps) in a model.
    Args:
        model (nn.Module): model over which to look for attentions.
        context (int or None): new temporary context value.

    ..Note:: this is not a context manager but a plain function changing the context forever.
        Initially, it was a context manager, but that led to interesting bugs when using
        activation checkpointing, with the context being inconsistent between the forward
        and backward.
    """
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context


class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    seq_start: torch.Tensor
    seq_len: torch.Tensor
    kv_padding: int

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> 'KVCacheResult':
        B, T, H, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, T, H)
        keys = keys.view(1, B * T, H, D)
        values = values.view(1, B * T, H, -1)
        seq_start = torch.arange(0, (B + 1) * T, T, device=keys.device, dtype=torch.int)
        seq_len = torch.full((B,), T, device=keys.device, dtype=torch.int)
        return KVCacheResult(keys, values, seq_start, seq_len, T)


class GraphableKVCache:
    """Efficient streaming KVCache to be compatible with Cuda Graph.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads in the attention.
        dim_per_head (int): Dimension per head.
        context (int, optional): Context size for the attention, if None, will grow exponentially,
            otherwise will use a fix allocation with a bit overhead.
        growth (float): Growth factor for the exponential growth, fraction of overhead when context is not None.
        initial_size (int): Initial size of the cache, used only when context is None.
        device (torch.device): Device on which to initialize the cache.
        dtype (torch.dtype): dtype to use for the cache.
        cache (torch.Tensor, optional): Initial cache, if provided. Shouldn't be used directly,
            use `clone()` instead.
        current_end (int): Current end of the cache, used only when cache is provided. Shouldn't be used directly,
            use `clone()` instead.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int | None = None,
        context: int | None = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        if context is None:
            assert capacity is not None, "Need at least context or capacity."
        else:
            if capacity is None:
                capacity = int(context * (1.2))
            else:
                assert capacity >= context
        assert capacity is not None
        self.context = context
        self.capacity = capacity
        self.context = context
        self.cache = torch.full(
            (2, batch_size, capacity, num_heads, dim_per_head),
            float("NaN"),
            device=device,
            dtype=dtype)
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1]
        B, T, H, D = k.shape
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        self.cache[0].index_copy_(1, indexes, k)
        self.cache[1].index_copy_(1, indexes, v)
        self.end_offset.add_(T)

        keys = self.cache[0]
        values = self.cache[1]
        seq_len = self.end_offset.expand(B)
        kv_padding = self.capacity

        seq_start = torch.arange(0, (B + 1) * kv_padding, kv_padding, device=indexes.device, dtype=torch.int)

        keys = keys.view(1, B * kv_padding, H, D)
        values = values.view(1, B * kv_padding, H, -1)
        return KVCacheResult(keys, values, seq_start.to(torch.int), seq_len.to(torch.int), kv_padding)


class StreamingMultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        mult = 1
        self.weights_per_step = weights_per_step
        if weights_per_step:
            mult = weights_per_step
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        B, T, H, D = k.shape
        if self._is_streaming:
            if 'kv_cache' not in self._streaming_state:
                if self.context is None:
                    capacity = self.weights_per_step or 1024
                else:
                    capacity = None
                self._streaming_state['kv_cache'] = GraphableKVCache(  # type: ignore
                    B, H, D,
                    context=self.context,
                    capacity=capacity,
                    device=k.device,
                    dtype=k.dtype)
            kv_cache: GraphableKVCache = self._streaming_state['kv_cache']  # type: ignore
            result = kv_cache.complete(k, v)
            return result
        else:
            return KVCacheResult.from_kv(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        streaming_offset: torch.Tensor
        streaming_offset_cpu: torch.Tensor
        if self._is_streaming:
            if 'offset' not in self._streaming_state:
                self._streaming_state['offset'] = torch.zeros(1, device=key.device, dtype=torch.long)
                self._streaming_state['offset_cpu'] = torch.zeros(1, dtype=torch.long)
            streaming_offset = self._streaming_state['offset']
            streaming_offset_cpu = self._streaming_state['offset_cpu']
        else:
            streaming_offset = torch.zeros(1, device=key.device, dtype=torch.long)
            streaming_offset_cpu = torch.zeros(1, dtype=torch.long)

        assert streaming_offset.device.type == "cuda"

        if self._is_streaming:
            assert self.causal, "Streaming only available for causal"

        if self.weights_per_step:
            projected = multi_linear(
                self.weights_per_step, self.in_proj_weight, query, int(streaming_offset_cpu.item())
            )
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
        packed = rearrange(projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads)
        q, k, v = ops.unbind(packed, dim=2)

        if self.rope:
            q, k = self.rope(q, k, offset=streaming_offset)

        kv_result = self._complete_kv(k, v)
        k, v = kv_result[:2]
        B, T, H, D = q.shape
        if self.causal:
            q = q.view(1, B * T, H, D)
            q_seqstart = torch.arange(0, T * (B + 1), T, device=q.device, dtype=torch.int)
            q_info = _SeqLenInfo(q_seqstart, T, T, list(range(0, T * B, B)))
            k_info = _PaddedSeqLenInfo(
                kv_result.seq_start,
                kv_result.kv_padding, 1,
                [None] * (B + 1),
                kv_result.seq_len, [None] * B,
                kv_result.kv_padding)
            attn_mask = BlockDiagonalCausalWithOffsetPaddedKeysMask(q_info, k_info)
        else:
            attn_mask = None
            assert "Not supported for now"
        dtype = q.dtype

        if q.device.type == "cpu":
            attn_bias = None
            if attn_mask is not None:
                attn_bias = attn_mask.materialize((q.shape[1], k.shape[1]))
            q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
            x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
            x = x.transpose(1, 2)
        else:
            x = ops.memory_efficient_attention(q, k, v, attn_mask, p=0)
            x = x.to(dtype)
        x = rearrange(x, "b t h d -> b t (h d)")
        if self.weights_per_step:
            x = multi_linear(
                self.weights_per_step, self.out_proj.weight, x, int(streaming_offset_cpu.item()),
            )
        else:
            x = self.out_proj(x)
        streaming_offset += T
        streaming_offset_cpu += T
        return x


class StreamingTransformerLayer(StreamingModule):
    """TransformerLayer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        norm (str): Normalization to use. Currently, only 'layer_norm' is supported.
        layer_scale (float, optional): If not None, LayerScale will be used with the given value as initial scale.
        gating (str): if provided, replaces FFN with special gating, like GLU, GSiGLU etc.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        skip_self_attn: If true, skips the self attention module and the norm
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                **attn_kwargs,
                **factory_kwargs,
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {weights_per_step}"
            )
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        offset = 0
        if self._is_streaming:
            offset = int(self._streaming_state["offset"].item())
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                B, T, D = x.shape
                ys = []
                for t in range(T):
                    y = self.gating[offset + t](x[:, t : t + 1])
                    ys.append(y)
                update = torch.cat(ys, dim=1)
            else:
                update = self.gating(x)
        return x_orig + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor):
        if self._is_streaming:
            if "offset" not in self._streaming_state:
                self._streaming_state["offset"] = torch.tensor(0)

        x = self._sa_block(x)
        x = self._ff_block(x)
        if self._is_streaming:
            self._streaming_state["offset"] += x.shape[1]
        return x


class StreamingTransformer(StreamingModule):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, sin_rope, or none).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `StreamingTransformerLayer`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        if "offsets" in self._streaming_state:
            offsets = self._streaming_state["offsets"]
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if self._is_streaming:
            self._streaming_state["offsets"] = offsets + T

        return x


class ProjectedTransformer(nn.Module):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(self, x, *args, **kwargs):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys

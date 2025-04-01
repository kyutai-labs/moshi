# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, + CUDA Graphable.
Optimized for inference.

See `StreamingTransformer` for more information.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import typing as tp
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils.compile import no_compile
from ..utils import quantize
from ..utils.quantize import replace_linear_with_qlinear
from .gating import make_gating
from .rope import RotaryEmbedding
from .streaming import StreamingModule, StreamingContainer, State
from .lora import LoRALinear
from torch.utils.checkpoint import checkpoint as torch_checkpoint


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
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
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
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions)


class RingKVCache:
    """Efficient streaming KVCache to be compatible with Cuda Graph.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads in the attention.
        dim_per_head (int): Dimension per head.
        device (torch.device): Device on which to initialize the cache.
        dtype (torch.dtype): dtype to use for the cache.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self):
        self.end_offset.zero_()

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        assert T > 0
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )

        # end_index correspond to the actual index where the last value was written.
        last_offset = self.end_offset + T - 1
        end_index = last_offset % self.capacity
        delta = indexes - end_index

        # We know that if `index == end_index`, then we should output `self.end_offset`.
        # If `index = end_index - 1` we should output `self.end_offset - 1`
        # If `index = end_index - n` we should output `self.end_offset - n`
        # Now, for `index == end_index + 1` , we actually have the oldest entry in the cache,
        # so we should output `end_index + 1 - self.capacity`

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        self.end_offset.add_(T)
        invalid = indexes >= self.end_offset
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


def apply_weights_per_step(modules: nn.ModuleList, schedule: list[int] | None,
                           x: torch.Tensor, offset: int) -> torch.Tensor:
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        modules (nn.ModuleList): apply weights per step.
        schedule (list[int] or None): schedule for weight sharing.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """

    if len(modules) == 1:
        return modules[0](x)

    ys: list[torch.Tensor] = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        if schedule is not None:
            module_index = schedule[module_index]
        y = modules[module_index](x[:, t: t + 1])
        ys.append(y)
    out = torch.cat(ys, 1)
    return out


@dataclass
class _MHAState(State):
    kv_cache: RingKVCache
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        weights_per_step_schedule (list[int] | None): if provided, some steps will share weights when
            `weights_per_step` is True, e.g. step `I` will use weights `schedule[I]`.
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
        weights_per_step_schedule: list[int] | None = None,
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
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule

        out_dim = 3 * embed_dim
        mult = 1
        if weights_per_step:
            if weights_per_step_schedule:
                assert len(weights_per_step_schedule) == weights_per_step
                mult = max(weights_per_step_schedule) + 1
            else:
                mult = weights_per_step
        self.mult = mult

        # Split in one linear per step
        self.out_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )
        self.in_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )

        self._register_load_state_dict_pre_hook(StreamingMultiheadAttention._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            'in_proj_weight': 'in_projs.{i}.weight',
            'in_proj.weight': 'in_projs.{i}.weight',
            'in_proj.lora_A.weight': 'in_projs.{i}.lora_A.weight',
            'in_proj.lora_B.weight': 'in_projs.{i}.lora_B.weight',
            'out_proj.weight': 'out_projs.{i}.weight',
            'out_proj.lora_A.weight': 'out_projs.{i}.lora_A.weight',
            'out_proj.lora_B.weight': 'out_projs.{i}.lora_B.weight',
        }

        mult = module.mult
        # _scb suffix is for quantized data.
        for suffix in ['', '_scb']:
            for source, target in mappings.items():
                this_source = prefix + source + suffix
                if this_source in state_dict:
                    weight = state_dict[this_source]
                    _, *OD = weight.shape
                    weight = weight.view(mult, -1, *OD)
                    for i in range(mult):
                        this_target = prefix + target.format(i=i) + suffix
                        state_dict[this_target] = weight[i]
                    state_dict.pop(this_source)

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        if self.context is None:
            if self.weights_per_step:
                capacity = self.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create a streaming KVCache without a context to estimate capacity."
                )
        else:
            capacity = self.context

        in_proj = self.in_projs[0]
        if isinstance(in_proj, LoRALinear):
            device = in_proj.lora_A.weight.device
            dtype = in_proj.lora_A.weight.dtype
        elif isinstance(in_proj, nn.Linear):
            device = in_proj.weight.device
            dtype = in_proj.weight.dtype
        elif isinstance(in_proj, quantize.QLinear):
            device = in_proj.weight.device
            dtype = torch.float16
        else:
            raise RuntimeError(f"Unknown type {type(in_proj)} for linear.")

        dim_per_head = self.embed_dim // self.num_heads
        kv_cache = RingKVCache(
            batch_size, self.num_heads, dim_per_head, capacity, device, dtype
        )
        return _MHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        state = self._streaming_state
        T = query.shape[1]

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset
            offset_cpu = state.offset_cpu

        projected = apply_weights_per_step(
            self.in_projs, self.weights_per_step_schedule, query, offset_cpu)

        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(
                -1, 1
            )
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = apply_weights_per_step(
            self.out_projs, self.weights_per_step_schedule, x, offset_cpu)

        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x


@dataclass
class _LayerState(State):
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0


class StreamingTransformerLayer(StreamingModule[_LayerState]):
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
        weights_per_step_schedule (list[int] | None): if provided, some steps will share weights when
            `weights_per_step` is True, e.g. step `I` will use weights `schedule[I]`.
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
        weights_per_step_schedule: list[int] | None = None,
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
                weights_per_step_schedule=weights_per_step_schedule,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        num_weights = 1
        if weights_per_step is not None:
            num_weights = weights_per_step
            if weights_per_step_schedule is not None:
                assert len(weights_per_step_schedule) == weights_per_step
                num_weights = max(weights_per_step_schedule) + 1
        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == num_weights, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {num_weights}"
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
                    dim_feedforward = [dim_feedforward] * num_weights
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
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        offset = 0
        if state is not None:
            offset = state.offset_cpu
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                update = apply_weights_per_step(self.gating, self.weights_per_step_schedule, x, offset)
            else:
                update = self.gating(x)
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor):
        with ExitStack() as stack:
            if x.device.type != 'cuda':
                stack.enter_context(no_compile())
            x = self._sa_block(x)
            x = self._ff_block(x)
            state = self._streaming_state
            if state:
                state.offset_cpu += x.shape[1]
            return x


@dataclass
class _TransformerState(State):
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()


class StreamingTransformer(StreamingModule[_TransformerState]):
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
        quantize: bool = False,
        checkpointing: bool = False,
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

        self.checkpointing = checkpointing

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
            if quantize:
                # Quantizing layers one by one to avoid taking too much space during init.
                self.layers[-1].to(device=device, dtype=dtype)
                replace_linear_with_qlinear(self.layers[-1])

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        dtype_input = x.dtype
        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offset.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            if self.checkpointing:
                y = torch_checkpoint(
                    layer, x, *args, use_reentrant=False,
                    determinism_check='none',
                    preserve_rng_state=False,
                    **kwargs)
                assert isinstance(y, torch.Tensor)
                x = y
            else:
                x = layer(x, *args, **kwargs)

        if state is not None:
            state.offset.add_(T)
        return x.to(dtype_input)


class ProjectedTransformer(StreamingContainer):
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

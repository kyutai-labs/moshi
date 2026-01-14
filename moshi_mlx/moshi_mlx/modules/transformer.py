# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .kv_cache import KVCache, RotatingKVCache

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    causal: bool
    norm_first: bool
    bias_ff: bool
    bias_attn: bool
    layer_scale: float | None
    positional_embedding: str
    use_conv_block: bool
    cross_attention: bool
    conv_kernel_size: int
    use_conv_bias: bool
    gating: bool
    norm: str
    context: int
    max_period: int
    max_seq_len: int
    kv_repeat: int
    dim_feedforward: int
    conv_layout: bool

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, xs: mx.array) -> mx.array:
        return xs


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.scale = mx.ones(dim)

    def __call__(self, xs: mx.array) -> mx.array:
        return xs * self.scale


@dataclass
class LayerCache:
    self_attn: KVCache | RotatingKVCache
    cross_attn: tuple[mx.array, mx.array] | None = None

    def reset(self):
        self.self_attn.reset()
        self.cross_attn = None


class CrossAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)

    def __call__(
        self,
        xs: mx.array,
        cross_attention_src: mx.array,
        cache: LayerCache,
    ) -> mx.array:
        # TODO: Add some cross-attention kv caching.
        assert self.cfg.kv_repeat == 1, "only kv_repeat==1 is supported"

        b, t, hd = xs.shape
        qkv_w = self.in_proj.weight
        q = xs @ qkv_w[: self.cfg.d_model].T
        q = q.reshape(b, t, self.cfg.num_heads, self.cfg.head_dim).swapaxes(1, 2)

        if cache.cross_attn is None:
            b_kv, t_kv, hd_kv = cross_attention_src.shape
            assert b == b_kv
            assert hd == hd_kv
            assert "bias" not in self.in_proj
            k = cross_attention_src @ qkv_w[self.cfg.d_model : 2 * self.cfg.d_model].T
            k = k.reshape(b, t_kv, self.cfg.num_heads, self.cfg.head_dim).swapaxes(1, 2)
            v = cross_attention_src @ qkv_w[2 * self.cfg.d_model :].T
            v = v.reshape(b, t_kv, self.cfg.num_heads, self.cfg.head_dim).swapaxes(1, 2)
            cache.cross_attn = k, v
        else:
            k, v = cache.cross_attn

        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        xs = self.out_proj(xs)
        return xs


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope":
            self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.max_period)

    def __call__(
        self,
        xs: mx.array,
        cache: KVCache | RotatingKVCache,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert self.cfg.kv_repeat == 1, "only kv_repeat==1 is supported"

        b, t, hd = xs.shape
        qkv = self.in_proj(xs).reshape(b, t, 3, self.cfg.num_heads, self.cfg.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)

        k, v = cache.update_and_fetch(k, v)
        k_len = k.shape[2]
        k_target_len = t + min(self.cfg.context, k_len - t)
        # TODO(laurent): the trimming below is incorrect for RotatingKVCache.
        # https://github.com/kyutai-labs/delayed-streams-modeling/issues/106
        if k_target_len < k_len:
            k = k[:, :, k_len - k_target_len :]
            v = v[:, :, k_len - k_target_len :]

        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        xs = self.out_proj(xs)
        return xs


class MlpGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        if cfg.dim_feedforward == 4 * cfg.d_model:
            hidden = 11 * cfg.d_model // 4

        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias_ff)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.linear_in(xs)
        b, t, _ = xs.shape
        xs = xs.reshape(b, t, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class MlpNoGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias_ff)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.linear2(nn.gelu_approx(self.linear1(xs)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        assert not cfg.use_conv_block, "conv-block is not supported"
        if cfg.gating:
            self.gating = MlpGating(cfg)
        else:
            # TODO: Use a better name?
            self.gating = MlpNoGating(cfg)

        if cfg.norm == "layer_norm":
            self.norm1 = nn.LayerNorm(cfg.d_model, 1e-5)
            self.norm2 = nn.LayerNorm(cfg.d_model, 1e-5)
        elif cfg.norm == "rms_norm":
            self.norm1 = nn.RMSNorm(cfg.d_model, 1e-8)
            self.norm2 = nn.RMSNorm(cfg.d_model, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.norm}")

        if cfg.layer_scale is not None:
            self.layer_scale_1 = LayerScale(cfg.d_model)
            self.layer_scale_2 = LayerScale(cfg.d_model)
        else:
            self.layer_scale_1 = Id()
            self.layer_scale_2 = Id()
        self.self_attn = Attention(cfg)
        self.cfg = cfg

        if cfg.cross_attention:
            # Always use layer-norm for the cross-attention.
            self.norm_cross = nn.LayerNorm(cfg.d_model, 1e-5)
            self.cross_attention = CrossAttention(cfg)
        else:
            self.cross_attention = None

    def _cross_attention_block(
        self,
        x: mx.array,
        cache: LayerCache,
        cross_attention_src: mx.array,
    ) -> mx.array:
        assert self.cross_attention is not None
        x_orig = x
        x = self.norm_cross(x)
        update = self.cross_attention(x, cross_attention_src, cache)
        return x_orig + update

    def __call__(
        self,
        xs: mx.array,
        cache: LayerCache,
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        n1 = self.norm1(xs)
        n1 = self.self_attn(n1, cache=cache.self_attn)
        xs = xs + self.layer_scale_1(n1)
        if self.cross_attention is not None:
            assert cross_attention_src is not None
            xs = self._cross_attention_block(xs, cache, cross_attention_src)
        else:
            assert cross_attention_src is None
        xs = xs + self.layer_scale_2(self.gating(self.norm2(xs)))
        return xs


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg=cfg) for _ in range(cfg.num_layers)]

    def __call__(
        self,
        xs: mx.array,
        cache: list[LayerCache],
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c, cross_attention_src=cross_attention_src)
        return xs

    def make_cache(self) -> list[LayerCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        return [
            LayerCache(KVCache(head_dim=self.cfg.head_dim, n_kv_heads=num_kv_heads))
            for _ in self.layers
        ]

    def make_rot_cache(self) -> list[LayerCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        return [
            LayerCache(
                RotatingKVCache(
                    head_dim=self.cfg.head_dim,
                    n_kv_heads=num_kv_heads,
                    max_size=self.cfg.max_seq_len,
                )
            )
            for _ in self.layers
        ]


class ProjectedTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, input_dim: int, output_dims: list[int]):
        super().__init__()

        self.conv_layout = cfg.conv_layout
        self.transformer = Transformer(cfg)
        if input_dim == cfg.d_model:
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(input_dim, cfg.d_model, bias=False)

        output_projs = []
        for output_dim in output_dims:
            if output_dim == cfg.d_model:
                p = None
            else:
                p = nn.Linear(cfg.d_model, output_dim, bias=False)
            output_projs.append(p)
        self.output_projs = output_projs

    def __call__(
        self,
        xs: mx.array,
        cache: list[LayerCache],
        cross_attention_src: None | mx.array = None,
    ) -> list[mx.array]:
        if self.conv_layout:
            xs = xs.swapaxes(1, 2)
        if self.input_proj is not None:
            xs = self.input_proj(xs)
        xs = self.transformer(xs, cache=cache, cross_attention_src=cross_attention_src)
        outs = []
        for output_proj in self.output_projs:
            if output_proj is None:
                out = xs
            else:
                out = output_proj(xs)
            if self.conv_layout:
                out = out.swapaxes(1, 2)
            outs.append(out)
        return outs

    def make_cache(self) -> list[LayerCache]:
        return self.transformer.make_cache()

    def make_rot_cache(self) -> list[LayerCache]:
        return self.transformer.make_rot_cache()

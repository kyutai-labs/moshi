# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.kv_cache import KVCache, RotatingKVCache
from ..modules.transformer import Transformer, TransformerConfig
from ..utils import sampling


@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int


@dataclass
class LmConfig:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: list[int]

    @classmethod
    def from_dict(cls, data: dict) -> "LmConfig":
        transformer = TransformerConfig(
            d_model=data["dim"],
            num_heads=data["num_heads"],
            num_layers=data["num_layers"],
            dim_feedforward=4 * data["dim"],
            causal=data["causal"],
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=data["layer_scale"],
            context=data["context"],
            max_period=data["max_period"],
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding=data["positional_embedding"],
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        )
        depformer = DepFormerConfig(
            transformer=TransformerConfig(
                d_model=data["depformer_dim"],
                num_heads=data["depformer_num_heads"],
                num_layers=data["depformer_num_layers"],
                dim_feedforward=data["depformer_dim_feedforward"],
                causal=data["depformer_causal"],
                norm_first=True,
                bias_ff=False,
                bias_attn=data["depformer_layer_scale"],
                layer_scale=None,
                context=data["depformer_context"],
                max_period=data["depformer_max_period"],
                use_conv_block=False,
                use_conv_bias=True,
                cross_attention=False,
                gating=True,
                norm="rms_norm",
                positional_embedding=data["depformer_pos_emb"],
                conv_layout=False,
                conv_kernel_size=3,
                kv_repeat=1,
                max_seq_len=4096,
            ),
            num_slices=data["dep_q"],
        )
        return LmConfig(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,
            text_out_vocab_size=data["text_card"],
            audio_vocab_size=data["card"] + 1,
            audio_delays=data["delays"][1:],  # the first delay is for the text token.
            audio_codebooks=data["n_q"],
        )

    @property
    def audio_eos_token(self) -> int:
        return self.audio_vocab_size - 2

    @property
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1


class DepFormerSlice(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        cfg: TransformerConfig,
    ):
        super().__init__()

        dim = cfg.d_model
        self.emb = nn.Embedding(in_vocab_size, dim)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        self.transformer = Transformer(cfg)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")


class DepFormer(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        self.slices: list[DepFormerSlice] = []
        for slice_idx in range(cfg.depformer.num_slices):
            in_vs = cfg.text_in_vocab_size if slice_idx == 0 else cfg.audio_vocab_size
            slice = DepFormerSlice(
                in_vs,
                cfg.audio_vocab_size - 1,
                main_transformer_dim=cfg.transformer.d_model,
                cfg=cfg.depformer.transformer,
            )
            self.slices.append(slice)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def sample(
        self,
        main_transformer_out: mx.array,
        step_idx: int,
        sampler: sampling.Sampler,
        text_token: mx.array,
        cache: list[KVCache] | list[RotatingKVCache],
    ) -> mx.array:
        tokens = []
        last_token = text_token
        # The cache is shared between the depformer slices but not persisted between sample calls.
        for c in cache:
            c.reset()
        for slice_idx, slice in enumerate(self.slices):
            last_token = (
                last_token if step_idx > 0 or slice_idx in (0, 1, 9) else mx.array(2048)
            )
            xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
            xs = slice.transformer(xs, cache=cache)
            logits = slice.linear_out(xs)
            last_token, _ = sampler(logits[0])
            tokens.append(last_token)
        tokens = mx.concatenate(tokens)
        return tokens


class Lm(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model
        self.transformer: Transformer = Transformer(cfg.transformer)
        self.depformer: DepFormer = DepFormer(cfg)
        self.text_emb = nn.Embedding(cfg.text_in_vocab_size, dim)
        self.cfg: LmConfig = cfg

        if cfg.transformer.norm == "layer_norm":
            self.out_norm = nn.LayerNorm(dim, 1e-5)
        elif cfg.transformer.norm == "rms_norm":
            self.out_norm = nn.RMSNorm(dim, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.transformer.norm}")

        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [
            nn.Embedding(cfg.audio_vocab_size, dim) for _ in range(cfg.audio_codebooks)
        ]
        self.transformer_cache: list[RotatingKVCache] = (
            self.transformer.make_rot_cache()
        )

        if len(self.depformer.slices) > 0:
            self.depformer_cache: list[KVCache] = self.depformer.slices[
                0
            ].transformer.make_cache()
        else:
            self.depformer_cache = []

    def __call__(
        self,
        token_ids: mx.array,
    ) -> mx.array:
        # Note that this does not apply the depformer.
        xs = self.text_emb(token_ids)
        transformer_out = self.transformer(xs, cache=self.transformer_cache)
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return text_logits

    def sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        step_idx: int,
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
    ) -> tuple[mx.array, mx.array]:
        xs = self.text_emb(text_token_ids)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            xs = xs + emb(token_ids)
        transformer_out = self.transformer(xs, cache=self.transformer_cache)
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        text_token, _ = text_sampler(text_logits[:, 0])
        audio_tokens = self.depformer.sample(
            transformer_out,
            step_idx,
            audio_sampler,
            text_token,
            self.depformer_cache,
        )
        return text_token, audio_tokens

    def warmup(self):
        text, audio = self.sample(
            mx.array([[self.cfg.text_out_vocab_size]]),
            [mx.array([[0]])] * 8,
            0,
            text_sampler=sampling.Sampler(),
            audio_sampler=sampling.Sampler(),
        )
        if text.sum().item() == 42:
            raise ValueError(42)
        if audio.sum().item() == 42:
            raise ValueError(42)
        for c in self.transformer_cache:
            c.reset()


def config1b_202412() -> LmConfig:
    transformer = TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_layers=16,
        dim_feedforward=2048 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48001,
        text_out_vocab_size=48000,
        audio_codebooks=16,
        audio_delays=([0] + [2] * 7) * 2,
    )


def config1b_202412_16rvq() -> LmConfig:
    transformer = TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_layers=16,
        dim_feedforward=2048 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=16,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=16,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48001,
        text_out_vocab_size=48000,
        audio_codebooks=32,
        audio_delays=([0] + [2] * 15) * 2,
    )


def config_v0_1() -> LmConfig:
    transformer = TransformerConfig(
        d_model=4096,
        num_heads=32,
        num_layers=32,
        dim_feedforward=4096 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=10000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=32001,
        text_out_vocab_size=32000,
        audio_codebooks=16,
        audio_delays=([0] + [1] * 7) * 2,
    )


def config_helium_1_preview_2b() -> LmConfig:
    transformer = TransformerConfig(
        d_model=2560,
        num_heads=20,
        num_layers=24,
        dim_feedforward=2560 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=4096,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=transformer,
        num_slices=0,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48000,
        text_out_vocab_size=48000,
        audio_codebooks=0,
        audio_delays=[],
    )

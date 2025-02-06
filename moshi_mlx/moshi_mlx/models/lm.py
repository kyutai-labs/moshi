# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.conditioner import LutConditionerConfig, ConditionProvider, ConditionTensor
from ..modules.kv_cache import KVCache, RotatingKVCache
from ..modules.transformer import Transformer, TransformerConfig
from ..utils import sampling


@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int
    weights_per_step_schedule: list[int] | None = None
    low_rank_embeddings: int | None = None


@dataclass
class LmConfig:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: list[int]
    conditioners: dict[str, LutConditionerConfig]

    @property
    def generated_codebooks(self):
        if self.depformer is None:
            return 0
        return self.depformer.num_slices

    @property
    def other_codebooks(self):
        return self.audio_codebooks - self.generated_codebooks

    @classmethod
    def from_config_dict(cls, data: dict) -> "LmConfig":
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
            weights_per_step_schedule=data.get("depformer_weights_per_step_schedule", None),
            low_rank_embeddings=data.get("depformer_low_rank_embeddings", None)
        )
        conditioners = {}
        if "conditioners" in data:
            for _name, _cfg in data["conditioners"].items():
                if _cfg["type"] != "lut":
                    raise ValueError(f"unsupported conditioner type {_cfg['type']}")
                _cfg = _cfg["lut"]
                _cfg = LutConditionerConfig(
                    n_bins=_cfg["n_bins"],
                    dim=_cfg["dim"],
                    tokenizer=_cfg["tokenizer"],
                    possible_values=_cfg["possible_values"],
                )
                conditioners[_name] = _cfg
        return LmConfig(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,
            text_out_vocab_size=data["text_card"],
            audio_vocab_size=data["card"] + 1,
            audio_delays=data["delays"][1:],  # the first delay is for the text token.
            audio_codebooks=data["n_q"],
            conditioners=conditioners,
        )

    @property
    def audio_eos_token(self) -> int:
        return self.audio_vocab_size - 2

    @property
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1


class LowRankEmbeddings(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        dim: int,
        low_rank_dim: int,
    ):
        super().__init__()
        scale = (1.0 / low_rank_dim) ** 0.5
        self.weight = mx.random.normal(shape=(in_vocab_size, low_rank_dim), scale=scale)
        self.low_rank = nn.Linear(low_rank_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.low_rank(self.weight[x])


class DepFormerSlice(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        cfg: DepFormerConfig,
    ):
        super().__init__()

        dim = cfg.transformer.d_model
        if cfg.low_rank_embeddings is None:
            self.emb = nn.Embedding(in_vocab_size, dim)
        else:
            self.emb = LowRankEmbeddings(in_vocab_size, dim, cfg.low_rank_embeddings)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        self.transformer = Transformer(cfg.transformer)

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
                cfg=cfg.depformer,
            )
            self.slices.append(slice)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def sample(
        self,
        main_transformer_out: mx.array,
        sampler: sampling.Sampler,
        text_token: mx.array,
        cache: list[KVCache] | list[RotatingKVCache],
        cfg_coef: float = 1.,
    ) -> mx.array:
        tokens = []
        last_token = text_token
        # The cache is shared between the depformer slices but not persisted between sample calls.
        for c in cache:
            c.reset()
        for slice in self.slices:
            # The 2048 tokens should be teacher forced on the first slices. However as delays
            # are non-decreasing in the number of slices, this is actually not necessary as
            # the generated tokens will end up not being used.
            last_token = last_token.reshape(1, 1)

            if cfg_coef != 1:
                last_token = mx.tile(last_token, (2, 1))
            xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
            xs = slice.transformer(xs, cache=cache)
            logits = slice.linear_out(xs)
            if cfg_coef != 1:
                l1, l2 = logits.split(2, axis=0)
                logits = cfg_coef * l1 - (cfg_coef - 1) * l2

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

        if len(cfg.conditioners) > 0:
            self.condition_provider = ConditionProvider(cfg.transformer.d_model, cfg.conditioners)
        else:
            self.condition_provider = None

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
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        ct: ConditionTensor | None = None,
        cfg_coef: float = 1.,
    ) -> tuple[mx.array, mx.array]:
        xs = self.text_emb(text_token_ids)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            xs = xs + emb(token_ids)
        if ct is not None:
            xs = xs + ct.tensor

        if cfg_coef != 1:
            xs = mx.tile(xs, (2, 1, 1))
        transformer_out = self.transformer(xs, cache=self.transformer_cache)
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        if cfg_coef != 1:
            l1, l2 = text_logits.split(2, axis=0)
            text_logits = cfg_coef * l1 - (cfg_coef - 1) * l2

        text_token, _ = text_sampler(text_logits[:, 0])
        audio_tokens = self.depformer.sample(
            transformer_out,
            audio_sampler,
            text_token,
            self.depformer_cache,
            cfg_coef=cfg_coef,
        )
        return text_token, audio_tokens

    def warmup(self, ct: ConditionTensor | None = None):
        text, audio = self.sample(
            mx.array([[self.cfg.text_out_vocab_size]]),
            [mx.array([[0]])] * self.cfg.other_codebooks,
            text_sampler=sampling.Sampler(),
            audio_sampler=sampling.Sampler(),
            ct=ct,
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
        conditioners={},
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
        conditioners={},
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
        conditioners={},
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
        conditioners={},
    )

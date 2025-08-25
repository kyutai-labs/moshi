# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.conditioner import (
    ConditionProvider,
    ConditionTensor,
    LutConditionerConfig,
    TensorConditionerConfig,
)
from ..modules.transformer import LayerCache, Transformer, TransformerConfig
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
    conditioners: dict[str, LutConditionerConfig | TensorConditionerConfig]
    demux_second_stream: bool = False
    extra_heads_num_heads: int = 0
    extra_heads_dim: int = 6

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
            cross_attention=data.get("cross_attention", False),
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
                causal=data.get("depformer_causal", True),
                norm_first=True,
                bias_ff=False,
                bias_attn=data.get("depformer_layer_scale", False),
                layer_scale=None,
                context=data.get("depformer_context", data["dep_q"]),
                max_period=data.get("depformer_max_period", 8),
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
            weights_per_step_schedule=data.get(
                "depformer_weights_per_step_schedule", None
            ),
            low_rank_embeddings=data.get("depformer_low_rank_embeddings", None),
        )
        conditioners = {}
        if "conditioners" in data:
            for _name, _cfg in data["conditioners"].items():
                if _cfg["type"] == "lut":
                    _cfg = _cfg["lut"]
                    _cfg = LutConditionerConfig(
                        n_bins=_cfg["n_bins"],
                        dim=_cfg["dim"],
                        tokenizer=_cfg["tokenizer"],
                        possible_values=_cfg["possible_values"],
                    )
                elif _cfg["type"] == "tensor":
                    _cfg = _cfg["tensor"]
                    _cfg = TensorConditionerConfig(
                        dim=_cfg["dim"],
                    )
                else:
                    raise ValueError(f"unsupported conditioner type {_cfg['type']}")
                conditioners[_name] = _cfg
        return LmConfig(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,
            text_out_vocab_size=data["text_card"],
            audio_vocab_size=data["card"] + 1,
            audio_delays=data["delays"][1:],  # the first delay is for the text token.
            audio_codebooks=data["n_q"],
            demux_second_stream=data.get("demux_second_stream", False),
            conditioners=conditioners,
            extra_heads_dim=data.get("extra_heads_dim", 6),
            extra_heads_num_heads=data.get("extra_heads_num_heads", 0),
        )

    @property
    def audio_eos_token(self) -> int:
        return self.audio_vocab_size - 2

    @property
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
        low_rank (int | None): if provided, uses low rank embedding with a linear layer to reach
            the desired dimension. Quite efficient for reducing the number of weights for very large vocabs.
        lr (float or None): learning rate to use, only valid if the `make_optim_group()` method is used.
        demux_second_stream (bool): input tokens can be the cartesian product of the vocab size,
            and they will be demuxed, e.g. `(tok2 * card + tok1)`. In that case the same embedding
            is used with different linear matrices.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        zero_idx: int = -1,
        low_rank: int | None = None,
        demux_second_stream: bool = False,
    ):
        super().__init__(num_embeddings, low_rank or embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.num_embeddings = num_embeddings
        self.zero_idx = zero_idx
        self.low_rank = None
        if low_rank is not None:
            self.low_rank = nn.Linear(low_rank, embedding_dim, bias=False)

        self.demux_second_stream = demux_second_stream
        assert self.zero_idx == -1, (
            "When demuxing a second stream, zero_idx must be -1."
        )
        if self.demux_second_stream:
            self.out1 = nn.Linear(low_rank or embedding_dim, embedding_dim, bias=False)
            self.out2 = nn.Linear(low_rank or embedding_dim, embedding_dim, bias=False)

    def __call__(self, input: mx.array) -> mx.array:
        is_zero = input == self.zero_idx
        zero = mx.zeros(1, dtype=input.dtype)
        input = mx.maximum(input, 0)
        if self.demux_second_stream:
            left = input % self.num_embeddings
            right = input // self.num_embeddings
            # Right is itself between [-1, ..., card - 1], with -1 being the zero value.
            right = right - 1
            left = self.weight[left]
            right_zero = (right < 0)[..., None]
            right = mx.maximum(right, 0)
            right = self.weight[right]
            y = self.out1(left) + mx.where(right_zero, zero, self.out2(right))
            y = mx.where(is_zero[..., None], zero, y)
        else:
            y = self.weight[input]
            y = mx.where(is_zero[..., None], zero, y)
            if self.low_rank is not None:
                y = self.low_rank(y)
        return y


class DepFormerSlice(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        demux_second_stream: bool,
        cfg: DepFormerConfig,
    ):
        super().__init__()

        dim = cfg.transformer.d_model
        self.emb = ScaledEmbedding(
            in_vocab_size,
            dim,
            low_rank=cfg.low_rank_embeddings,
            demux_second_stream=demux_second_stream,
        )
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
                demux_second_stream=slice_idx == 0 and cfg.demux_second_stream,
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
        cache: list[LayerCache],
        cfg_coef: float = 1.0,
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

            if cfg_coef != 1:
                last_token = mx.tile(last_token, (2, 1))
            xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
            xs = slice.transformer(xs, cache=cache)
            logits = slice.linear_out(xs)
            if cfg_coef != 1:
                l1, l2 = logits.split(2, axis=0)
                logits = cfg_coef * l1 - (cfg_coef - 1) * l2

            last_token, _ = sampler(logits)
            tokens.append(last_token)
        tokens = mx.stack(tokens, axis=1)
        return tokens


class Lm(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model
        self.transformer: Transformer = Transformer(cfg.transformer)
        self.depformer: DepFormer = DepFormer(cfg)
        self.text_emb = ScaledEmbedding(
            cfg.text_in_vocab_size, dim, demux_second_stream=cfg.demux_second_stream
        )
        self.cfg: LmConfig = cfg

        if cfg.transformer.norm == "layer_norm":
            self.out_norm = nn.LayerNorm(dim, 1e-5)
        elif cfg.transformer.norm == "rms_norm":
            self.out_norm = nn.RMSNorm(dim, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.transformer.norm}")

        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [
            ScaledEmbedding(cfg.audio_vocab_size, dim)
            for _ in range(cfg.audio_codebooks)
        ]
        self.extra_heads = [
            nn.Linear(dim, cfg.extra_heads_dim, bias=False)
            for _ in range(cfg.extra_heads_num_heads)
        ]
        self.transformer_cache: list[LayerCache] = self.transformer.make_rot_cache()

        if len(self.depformer.slices) > 0:
            self.depformer_cache: list[LayerCache] = self.depformer.slices[
                0
            ].transformer.make_cache()
        else:
            self.depformer_cache = []

        if len(cfg.conditioners) > 0:
            self.condition_provider = ConditionProvider(
                cfg.transformer.d_model, cfg.conditioners
            )
        else:
            self.condition_provider = None

    def load_pytorch_weights(
        self,
        file: str,
        lm_config: LmConfig,
        strict: bool = True,
    ) -> nn.Module:
        pth_t = mx.load(file)
        depformer_chunks = lm_config.depformer.num_slices
        if lm_config.depformer.weights_per_step_schedule is not None:
            depformer_chunks = max(lm_config.depformer.weights_per_step_schedule) + 1

        mlx_t = {}
        mlx_t["out_norm.weight"] = pth_t["out_norm.alpha"][0, 0]
        for name in [
            "text_emb.out1.weight",
            "text_emb.out2.weight",
            "text_emb.weight",
            "text_linear.weight",
        ]:
            if name in pth_t:
                mlx_t[name] = pth_t[name]
        for cb_idx in range(lm_config.audio_codebooks):
            mlx_t[f"audio_embs.{cb_idx}.weight"] = pth_t[f"emb.{cb_idx}.weight"]
        for k, v in sorted(pth_t.items()):
            if k.startswith("transformer"):
                if k.endswith(".alpha"):
                    v = v[0, 0]
                k = k.replace(".alpha", ".weight")
                k = k.replace(".in_proj_weight", ".in_proj.weight")
                mlx_t[k] = v
            if k.startswith("condition_provider.") or k.startswith("extra_heads."):
                mlx_t[k] = v

        for slice_idx in range(lm_config.depformer.num_slices):
            pth_idx = slice_idx
            if lm_config.depformer.weights_per_step_schedule is not None:
                pth_idx = lm_config.depformer.weights_per_step_schedule[slice_idx]
            slice_p = f"depformer.slices.{slice_idx}"
            mlx_t[f"{slice_p}.linear_in.weight"] = pth_t[
                f"depformer_in.{pth_idx}.weight"
            ]
            mlx_t[f"{slice_p}.linear_out.weight"] = pth_t[f"linears.{slice_idx}.weight"]
            if slice_idx == 0:
                mlx_t[f"{slice_p}.emb.weight"] = pth_t["depformer_text_emb.weight"]
                for _n in ["low_rank", "out1", "out2"]:
                    if f"depformer_text_emb.{_n}.weight" in pth_t:
                        mlx_t[f"{slice_p}.emb.{_n}.weight"] = pth_t[
                            f"depformer_text_emb.{_n}.weight"
                        ]
            else:
                mlx_t[f"{slice_p}.emb.weight"] = pth_t[
                    f"depformer_emb.{slice_idx - 1}.weight"
                ]
                if f"depformer_emb.{slice_idx - 1}.low_rank.weight" in pth_t:
                    mlx_t[f"{slice_p}.emb.low_rank.weight"] = pth_t[
                        f"depformer_emb.{slice_idx - 1}.low_rank.weight"
                    ]
            for layer_idx in range(lm_config.depformer.transformer.num_layers):
                p = f"{slice_p}.transformer.layers.{layer_idx}"
                mlx_t[f"{p}.norm1.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.norm1.alpha"
                ][0, 0]
                mlx_t[f"{p}.norm2.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.norm2.alpha"
                ][0, 0]
                mlx_t[f"{p}.gating.linear_in.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_in.weight"
                ]
                mlx_t[f"{p}.gating.linear_out.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_out.weight"
                ]
                mlx_t[f"{p}.self_attn.in_proj.weight"] = mx.split(
                    pth_t[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"],
                    depformer_chunks,
                )[pth_idx]
                mlx_t[f"{p}.self_attn.out_proj.weight"] = mx.split(
                    pth_t[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"],
                    depformer_chunks,
                )[pth_idx]
        return self.load_weights(list(mlx_t.items()), strict=strict)

    @property
    def n_q(self) -> int:
        return self.cfg.audio_codebooks

    @property
    def dep_q(self) -> int:
        return self.cfg.depformer.num_slices

    @property
    def audio_offset(self) -> int:
        return 1

    @property
    def delays(self) -> list[int]:
        return self.cfg.audio_delays

    def forward_text(
        self,
        token_ids: mx.array,
        cross_attention_src: None | mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        # Note that this does not apply the depformer.
        xs = self.text_emb(token_ids)
        transformer_out = self.transformer(
            xs, cache=self.transformer_cache, cross_attention_src=cross_attention_src
        )
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return (transformer_out, text_logits)

    def __call__(
        self,
        token_ids: mx.array,
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        # Note that this does not apply the depformer.
        xs = self.text_emb(token_ids)
        transformer_out = self.transformer(
            xs, cache=self.transformer_cache, cross_attention_src=cross_attention_src
        )
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return text_logits

    def _sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        ct: ConditionTensor | None = None,
        cross_attention_src: None | mx.array = None,
        cfg_coef: float = 1.0,
        on_text_hook=None,
        on_audio_hook=None,
    ) -> tuple[mx.array, mx.array | None, mx.array]:
        xs = self.text_emb(text_token_ids)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            _emb = emb(token_ids)
            _emb = _emb.transpose(1, 0, 2)
            xs = xs + _emb
        if ct is not None:
            xs = xs + mx.expand_dims(ct.tensor, axis=1)
        if cfg_coef != 1:
            xs = mx.tile(xs, (2, 1, 1))
        transformer_out = self.transformer(
            xs,
            cache=self.transformer_cache,
            cross_attention_src=cross_attention_src,
        )
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        if cfg_coef != 1:
            l1, l2 = text_logits.split(2, axis=0)
            text_logits = cfg_coef * l1 - (cfg_coef - 1) * l2
        text_token, _ = text_sampler(text_logits)
        if on_text_hook is not None:
            on_text_hook(text_token)
        if len(self.depformer.slices) > 0:
            audio_tokens = self.depformer.sample(
                transformer_out,
                audio_sampler,
                text_token,
                self.depformer_cache,
                cfg_coef=cfg_coef,
            )
            if on_audio_hook is not None:
                on_audio_hook(audio_tokens)
        else:
            audio_tokens = None
        return text_token, audio_tokens, transformer_out

    def sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        ct: ConditionTensor | None = None,
        cross_attention_src: None | mx.array = None,
        cfg_coef: float = 1.0,
        on_text_hook=None,
        on_audio_hook=None,
    ) -> tuple[mx.array, mx.array | None]:
        text, audio, _ = self._sample(
            text_token_ids,
            audio_token_ids,
            text_sampler,
            audio_sampler,
            ct,
            cross_attention_src,
            cfg_coef,
            on_text_hook,
            on_audio_hook,
        )
        return text, audio

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
        if audio is not None and audio.sum().item() == 42:
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

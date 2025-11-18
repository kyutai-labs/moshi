# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules import (
    ConvDownsample1d,
    ConvTranspose1d,
    ConvTrUpsample1d,
    EuclideanCodebook,
    ProjectedTransformer,
    SeanetConfig,
    SeanetDecoder,
    SeanetEncoder,
    SplitResidualVectorQuantizer,
    TransformerConfig,
)


@dataclass
class MimiConfig:
    channels: int
    sample_rate: float
    frame_rate: float
    renormalize: bool
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int


def mimi_202407(num_codebooks: int) -> MimiConfig:
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=0.01,
        positional_embedding="rope",
        use_conv_bias=True,
        gating=False,
        norm="layer_norm",
        context=250,
        max_period=10000,
        max_seq_len=8192,
        kv_repeat=1,
        dim_feedforward=2048,
        conv_layout=True,
        use_conv_block=False,
        cross_attention=False,
        conv_kernel_size=3,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        renormalize=True,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks,
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.upsample = ConvTrUpsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        for c in self.encoder_cache:
            c.reset()

    def reset_all(self):
        self.reset_state()
        self.upsample.reset_state()
        self.downsample.reset_state()

    def encode(self, xs: mx.array) -> mx.array:
        self.encoder.reset_state()
        for c in self.encoder_cache:
            c.reset()
        xs = self.encoder(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample(xs)
        return self.quantizer.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        xs = self.quantizer.decode(xs)
        xs = self.upsample(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        return self.decoder(xs)

    def encode_step(self, xs: mx.array) -> mx.array:
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)
        xs = self.quantizer.encode(xs)
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        xs = self.quantizer.decode(xs)
        xs = self.upsample.step(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)
        return xs

    def warmup(self):
        pcm = mx.zeros((1, 1, 1920 * 4))
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.eval(pcm_out)

    @property
    def frame_rate(self) -> float:
        return self.cfg.frame_rate

    @property
    def sample_rate(self) -> float:
        return self.cfg.sample_rate

    def load_pytorch_weights(
        self,
        file: str,
        strict: bool = True,
    ) -> nn.Module:
        weights = []
        for k, v in mx.load(file).items():
            v: mx.array = v
            k: str = ".".join([s.removeprefix("_") for s in k.split(".")])
            if k.startswith("encoder.model."):
                k = k.replace("encoder.model.", "encoder.")
            if k.startswith("decoder.model."):
                k = k.replace("decoder.model.", "decoder.")
            if k.endswith(".in_proj_weight"):
                k = k.replace(".in_proj_weight", ".in_proj.weight")
            if k.endswith(".linear1.weight"):
                k = k.replace(".linear1.weight", ".gating.linear1.weight")
            if k.endswith(".linear2.weight"):
                k = k.replace(".linear2.weight", ".gating.linear2.weight")
            # Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
            for layerIdx, decoderIdx in enumerate([2, 5, 8, 11]):
                k = k.replace(
                    f"decoder.{decoderIdx}.", f"decoder.layers.{layerIdx}.upsample."
                )
                k = k.replace(
                    f"decoder.{decoderIdx + 1}.",
                    f"decoder.layers.{layerIdx}.residuals.0.",
                )
            for layerIdx, encoderIdx in enumerate([1, 4, 7, 10]):
                k = k.replace(
                    f"encoder.{encoderIdx}.", f"encoder.layers.{layerIdx}.residuals.0."
                )
                k = k.replace(
                    f"encoder.{encoderIdx + 2}.",
                    f"encoder.layers.{layerIdx}.downsample.",
                )

            k = k.replace("decoder.0.", "decoder.init_conv1d.")
            k = k.replace("decoder.14.", "decoder.final_conv1d.")
            k = k.replace("encoder.0.", "encoder.init_conv1d.")
            k = k.replace("encoder.14.", "encoder.final_conv1d.")
            k = k.replace(".block.1.", ".block.0.")
            k = k.replace(".block.3.", ".block.1.")

            # PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
            if (
                k.endswith(".conv.weight")
                or k.endswith(".output_proj.weight")
                or k.endswith(".input_proj.weight")
            ):
                v = v.swapaxes(-1, -2)
            # PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
            if k.endswith(".convtr.weight"):
                v = v.transpose(1, 2, 0)
            weights.append((k, v))
        m = self.load_weights(weights, strict=strict)

        def _filter_fn(module, name, _):
            if isinstance(module, EuclideanCodebook) and name == "initialized":
                module.update_in_place()
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        m.filter_and_map(_filter_fn)
        return m

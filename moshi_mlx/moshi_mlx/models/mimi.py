# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from ..modules import SeanetConfig, TransformerConfig


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

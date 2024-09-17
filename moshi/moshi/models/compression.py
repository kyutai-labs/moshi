# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models. In particular, provides the implementation
for Mimi. Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp

import torch
from torch import nn


from ..quantization import (
    QuantizedResult,
    BaseQuantizer,
    SplitResidualVectorQuantizer,
    ResidualVectorQuantizer,
)
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.streaming import StreamingModule, State
from ..utils.compile import no_compile, CUDAGraphed


logger = logging.getLogger()


class CompressionModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...


@dataclass
class _MimiState:
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None

    def reset(self):
        pass


class MimiModel(CompressionModel[_MimiState]):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
        torch_compile_encoder_decoder (bool): if True, uses torch.compile on the encoder / decoder.
            Deactivated by default for training as this is incompatible at the moment with weight norm.
            See https://github.com/pytorch/pytorch/issues/121902
            Also this seems to work well with 2.2.0, but completely fail with 2.4.0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        if freeze_quantizer:
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    def _init_streaming_state(self, batch_size: int) -> _MimiState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        graphed_tr_enc = None
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        return _MimiState(graphed_tr_enc, graphed_tr_dec)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    @property
    def _context_for_encoder_decoder(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}

        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
        # Checking that we have the proper length given the advertised frame rate.
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )

        q_res = self.quantizer(emb, self.frame_rate)
        emb = q_res.x
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = out
        q_res.metrics.update(extra_metrics)
        return q_res

    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        state = self._streaming_state
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        emb = self._to_framerate(emb)
        return emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x)
        codes = self.quantizer.encode(emb)
        return codes

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        state = self._streaming_state
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(emb)
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel[State]):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        return self.model.forward(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks

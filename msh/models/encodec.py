# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
import logging
import typing as tp

import einops
import torch
from torch import nn
from torch.nn import functional as F


from .. import quantization as qt
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.streaming import StreamingModule
from ..utils.compile import no_compile


logger = logging.getLogger()


class CompressionModel(ABC, StreamingModule):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> qt.QuantizedResult: ...

    @abstractmethod
    def encode(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
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


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

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
        renormalize (bool): Whether to renormalize the audio before running the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        mask_fn (nn.Module or None): optional mask function to apply to the latent space.
        mask_position (str): whether to apply the mask function after the encoder or after the quantizer.
        mae_pre (bool): whether to use a mean absolute error loss before the quantizer.
        mae_post (bool): whether to use a mean absolute error loss after the quantizer.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
        torch_compile_encoder_decoder (bool): if True, uses torch.compile on the encoder / decoder.
            Deactivated by default for training as this is incompatible at the moment with weight norm.
            See https://github.com/pytorch/pytorch/issues/121902

    ..Warning:: autocast to float16 for EnCodec will not use a grad scaler. This is because
        the loss balancer will cancel out the grad scaling, and lead to weird effects.
    """

    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: qt.BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        renormalize: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        mask_fn: tp.Optional[nn.Module] = None,
        mask_position: str = "after_encoder",
        mae_pre: bool = False,
        mae_post: bool = False,
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
        self.mae_pre = mae_pre
        self.mae_post = mae_post
        self.quantizer = quantizer
        self.frame_rate = frame_rate
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

        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, "Causal model does not support renormalize"
        self.mask_fn = mask_fn
        self.mask_position = mask_position
        if self.mask_fn is not None:
            if self.mask_position not in [
                "after_encoder",
                "after_quantizer",
            ] and not self.mask_position.startswith("after_conv"):
                raise ValueError(
                    "mask_position should be in ['after_encoder', 'after_quantizer', 'after_conv_{idx}']"
                    f" but got {self.mask_position}."
                )
            if mask_position == "after_encoder":
                assert (
                    encoder_transformer is not None
                ), "Mask after encoder requires encoder_transformer"
        if self.mask_position.startswith("after_conv"):
            assert not (
                self.mae_post or self.mae_pre
            ), "MAE not supported with mask after conv."

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

    def preprocess(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(
        self, x: torch.Tensor, scale: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

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

    @staticmethod
    def _normalized_masked_mse(
        mae_estimate: torch.Tensor, unmasked: torch.Tensor, mask: torch.Tensor
    ):
        # mask from Gilbert masking is True where the information is available, we want to
        # focus on where information was masked.
        masked_positions = 1 - mask
        masked_positions = masked_positions.expand_as(mae_estimate)
        unmasked = unmasked.detach()
        mae_loss = F.mse_loss(
            mae_estimate * masked_positions, unmasked * masked_positions
        )
        mae_loss *= mae_estimate.numel() / (1e-8 + masked_positions.float().sum())
        mae_loss /= unmasked.var()
        return mae_loss

    @property
    def _context_for_encoder_decoder(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)
        extra_metrics: tp.Dict[str, torch.Tensor] = {}

        if self.freeze_quantizer:
            if isinstance(self.quantizer, qt.SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, qt.ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.mask_fn is not None and self.mask_position == "after_encoder":
            unmasked = emb.detach()
            emb, mask = self.mask_fn(emb)
        if self.encoder_transformer is not None:
            outs = self.encoder_transformer(emb)
            if self.mae_pre:
                assert (
                    self.mask_position == "after_encoder"
                ), "Cannot MAE pre-quantizer with mask after quantizer."
                emb, mae_estimate = outs
                mae_loss = EncodecModel._normalized_masked_mse(
                    mae_estimate, unmasked, mask
                )
                extra_metrics["mae_pre"] = mae_loss
            else:
                (emb,) = outs
        unquantized_before_resample = emb
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
        if self.mask_fn is not None and self.mask_position == "after_quantizer":
            unmasked = unquantized_before_resample.detach()
            emb, mask = self.mask_fn(emb)
        if self.decoder_transformer is not None:
            outs = self.decoder_transformer(emb)
            if self.mae_post:
                emb, mae_estimate = outs
                mae_loss = EncodecModel._normalized_masked_mse(
                    mae_estimate, unmasked, mask
                )
                extra_metrics["mae_post"] = mae_loss
            else:
                (emb,) = outs

        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)
        q_res.metrics.update(extra_metrics)
        return q_res

    def _encode_to_unquantized_latent(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings, and scale for audio renormalization.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        x, scale = self.preprocess(x)
        if scale is not None and self._is_streaming:
            raise ValueError("cannot normalize in streaming mode")
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            emb = self.encoder_transformer(emb)[0]
        emb = self._to_framerate(emb)
        return emb, scale

    def encode(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalization.
        """
        emb, scale = self._encode_to_unquantized_latent(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb, _ = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            emb = self.decoder_transformer(emb)[0]
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        return self.model.forward(x)

    def encode(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        return self.model.decode(codes, scale)

    def decode_latent(self, codes: torch.Tensor):
        self.model.decode_latent(codes)

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

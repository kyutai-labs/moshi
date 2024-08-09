import functools
import typing as tp

import einops
import torch
from torch.nn import functional as F
from transformers import WavLMModel

from .base import _BaseWaveformConditioner, WavCondition, ConditionType
from ..data.audio_utils import convert_audio
from ..models.demucs import get_demucs
from ..solvers.compression import CompressionSolver
from ..utils.utils import length_to_mask
from ..modules.transformer import ProjectedTransformer


class WaveformConditioner(_BaseWaveformConditioner[WavCondition]):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """

    def __init__(self, finetune: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.finetune = finetune

    def prepare(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        return WavCondition(wav.to(self.device), length.to(self.device), sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a dense embedding.

        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
                In particular `x.wav` contains the waveform in the `[B, C, T']` format.
        Returns:
            torch.Tensor: conditioning tensor (without masking) of shape `[B, T, dim]`.

        """
        raise NotImplementedError()

    def _get_frame_rate(self) -> float:
        """Returns the frame rate of the embedding model."""
        raise NotImplementedError()

    def _get_mask(self, embeds: torch.Tensor, x: WavCondition) -> torch.Tensor:
        frame_rate = self._get_frame_rate()
        duration = x.length.float() / x.sample_rate
        length = (duration * frame_rate).int()
        return length_to_mask(length, max_len=embeds.shape[1])

    def _get_condition(self, x: WavCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.

        In general you should not override directly this function but instead override
        the `_get_wav_embedding` method.

        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
                `(cond, mask)` with `cond` being of shape `[B, T, dim]` and mask `[B, T]`.

        """
        if x.is_nullified:
            return ConditionType(
                torch.zeros(len(x.wav), 1, self.dim, device=self.device),
                torch.zeros(len(x.wav), 1, device=self.device, dtype=torch.bool))
        with torch.set_grad_enabled(self.finetune):
            embeds = self._get_wav_embedding(x)
        mask = self._get_mask(embeds, x)
        return ConditionType(embeds, mask)


class NoiseLevelConditioner(WaveformConditioner):
    """Provide a conditioning based on the noise level in the input waveform.

    Args:
        output_dim (int): Output dimension.
        window_duration (float): Duration of the window used to compute the noise level.
        interpolate (bool): if True, will average the embeddings over the time dimension.
            Then, the conditioning is sequence-wise,
            otherwise we can control the conditioning per hop_duration chunks.
        levels (list[float]): should be a list of dB relative volume compared to the mix
            (will always be negative). Will be sorted in ascending order. Level 0
            will be anything lower than `levels[0]`, then level is above that
            but lower than `levels[1]` etc.
        floor (float): if the mix is lower that this level (in dB), then the mix is considered
            silent and the relative volume of the noise is not considered. A special token
            indicates silence.
        device (tp.Union[torch.device, str]): Device.
    """
    def __init__(
            self,
            output_dim: int,
            window_duration: float = 4., interpolate: bool = False,
            levels: tp.List[float] = [-24, -12, -6], floor: float = -40,
            **kwargs):
        super().__init__(dim=output_dim, output_dim=output_dim, force_linear=False, **kwargs)
        self.__dict__['demucs'] = get_demucs().to(self.device)
        self.window_duration = window_duration
        self.hop_duration = window_duration / 2
        self.interpolate = interpolate
        self.levels = sorted(levels)
        self.floor = floor
        self.embedding = torch.nn.Embedding(len(levels) + 2, self.dim)

    def _get_noise_level(self, x: WavCondition) -> torch.Tensor:
        """Returns the noise level as integers from 0 (low noise) to `len(levels)` (a lot of noise)
        and `len(levels) + 1` (mix is silent).
        """
        wav = convert_audio(x.wav, x.sample_rate, self.demucs.sample_rate, self.demucs.chin)
        window = int(self.window_duration * x.sample_rate)
        hop_length = int(self.hop_duration * x.sample_rate)
        padding = (window - hop_length) // 2

        if wav.shape[-1] < window:
            return torch.zeros(len(x.wav), 1, dtype=torch.long, device=self.device)

        denoised = self.demucs(wav)
        noise = wav - denoised

        def _volume(x):
            avg = F.avg_pool1d(x.pow(2), window, hop_length, padding)[:, 0]
            return (10 * torch.log10(avg))

        # Both volumes are [B, F]
        volume_ref = _volume(wav)
        noise_relative_volume = _volume(noise) - volume_ref

        values = torch.zeros(*volume_ref.shape, dtype=torch.long, device=self.device)

        for idx, level in enumerate(self.levels):
            values[noise_relative_volume > level] = idx + 1

        mix_is_silent = volume_ref <= self.floor
        values[mix_is_silent] = len(self.levels) + 1

        return values

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        level = self._get_noise_level(x)
        out = self.embedding(level)
        if self.interpolate:
            out = out.mean(dim=1, keepdim=True)
        return out

    def _get_frame_rate(self) -> float:
        return 1 / self.hop_duration

    def _get_mask(self, embeds: torch.Tensor, x: WavCondition) -> torch.Tensor:
        if self.interpolate:
            return torch.ones(len(x.wav), 1, device=self.device, dtype=torch.bool)
        else:
            return super()._get_mask(embeds, x)


class EncodecConditioner(WaveformConditioner):
    """Embeds a waveform into the latent space of an Encodec model.

    Args:
        compression_model_checkpoint (str): Path to the compression model.
        quantize (bool): Whether to quantize the latent space or return unquantized embeddings.
        **kwargs (dict): Additional arguments to the WaveformConditioner class.
    """

    def __init__(self, compression_model_checkpoint: str, quantize: bool = False, **kwargs):
        compression_model = CompressionSolver.model_from_checkpoint(compression_model_checkpoint, kwargs["device"])
        super().__init__(dim=compression_model.dimension, **kwargs)
        if self.finetune:
            self.compression_model = compression_model
        else:
            self.__dict__['compression_model'] = compression_model
        self.quantize = quantize

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a quantized Encodec embedding.

        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
                In particular `x.wav` contains the waveform in the `[B, C, T']` format.
        Returns:
            torch.Tensor: conditioning tensor (without masking) of shape `[B, T, dim]`.

        """
        inputs, sample_rate = x.wav, x.sample_rate
        inputs = convert_audio(
            inputs, sample_rate, self.compression_model.sample_rate, self.compression_model.channels
        )
        embeds = self.compression_model.encode_to_latent(inputs, quantize=self.quantize)
        embeds = embeds.transpose(1, 2)
        return embeds

    def _get_frame_rate(self) -> float:
        """Returns the frame rate of the embedding model."""
        return self.compression_model.frame_rate


class WavLMConditioner(WaveformConditioner):
    """Embeds a waveform into the latent space of a WavLM model.

    Args:
        **kwargs (dict): Additional arguments to the WaveformConditioner class.
    """
    def __init__(self, **kwargs):

        super().__init__(dim=1024, **kwargs)
        if self.finetune:
            self.wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(self.device)
        else:
            self.__dict__['wavlm_model'] = WavLMModel.from_pretrained('microsoft/wavlm-large').to(self.device)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a quantized Encodec embedding.

        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
                In particular `x.wav` contains the waveform in the `[B, C, T']` format.
        Returns:
            torch.Tensor: conditioning tensor (without masking) of shape `[B, T, dim]`.

        """
        inputs, sample_rate = x.wav, x.sample_rate
        if inputs.shape[1] != 1:
            raise ValueError(f"WavLMConditioner expects 1 channel, got {inputs.shape[1]}.")
        inputs = convert_audio(inputs, sample_rate, 16_000, 1)
        inputs = torch.squeeze(inputs, dim=1)
        # We need to pad with 80 values along time to avoid a missing frame at the end.
        inputs = F.pad(inputs, (0, 80), mode='constant', value=0)
        return self.wavlm_model(inputs)[0]

    def _get_frame_rate(self) -> float:
        """Returns the frame rate of the embedding model."""
        return 50.


def wrap_conditioner_for_multi_speaker(conditioner_cls):
    """Wraps a conditioner to handle multiple speakers.

    This wrapper embeds each channel (speaker) separately and then reconcatenate them along the time axis.
    We do not use x.length to mask silence as it is not reliable for multi-speaker data. We instead use a max_pooling
    with the same framerate as the conditioner to mark as zeros frames that are entirely comprised of silence.
    """

    class MultiSpeakerWrapper(conditioner_cls):
        """Multi-speaker wrapper for audio conditioners.

        Args:
            transformer (dict or None): Kwargs to the ProjectedTransformer class, used on top of embeddings.
            interpolate (bool): If True, will average the embeddings over the time dimension after the Transformer.
            interpolate_stride (int): Stride for the avg pool interpolation. If zero, the time dimension is reduced
                to 1.
            **kwargs: Additional arguments to the conditioner class.
        """
        def __init__(self, transformer: dict | None, interpolate: bool = False, interpolate_stride: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.interpolate_fn: tp.Optional[tp.Callable[[torch.Tensor], torch.Tensor]] = None
            self.interpolate_stride = interpolate_stride
            if interpolate:
                if self.interpolate_stride > 0:
                    self.interpolate_fn = torch.nn.AvgPool2d(
                        kernel_size=(2 * interpolate_stride, 1),
                        stride=(interpolate_stride, 1),
                        padding=(interpolate_stride // 2, 0),
                    )
                else:
                    self.interpolate_fn = functools.partial(torch.mean, dim=2, keepdim=True)
            use_transformer = False
            apply_transformer_per_speaker = False
            if transformer is not None:
                use_transformer = transformer.pop("use", False)
                apply_transformer_per_speaker = transformer.pop("per_speaker", False)
            self.apply_transformer_per_speaker = apply_transformer_per_speaker
            self.transformer = None
            if use_transformer:
                assert transformer is not None
                self.transformer = ProjectedTransformer(
                    input_dimension=self.dim,
                    output_dimensions=(self.dim,),
                    **transformer,
                )

        def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
            inputs = x.wav
            B, C, T = inputs.shape
            # C is the number of speakers.
            inputs = einops.rearrange(inputs, "b c t -> (b c) 1 t")
            embeds = super()._get_wav_embedding(WavCondition(inputs, x.length, x.sample_rate, x.path, x.seek_time))
            if not self.apply_transformer_per_speaker:
                embeds = einops.rearrange(embeds, "(b c) t d -> b (c t) d", b=B, c=C)
            if self.transformer is not None:
                embeds, = self.transformer(embeds)
            if self.apply_transformer_per_speaker:
                embeds = einops.rearrange(embeds, "(b c) t d -> b (c t) d", b=B, c=C)
            if self.interpolate_fn:
                embeds = einops.rearrange(embeds, "b (c t) d -> b c t d", b=B, c=C)
                embeds = self.interpolate_fn(embeds)
                embeds = einops.rearrange(embeds, "b c t d -> b (c t) d")
            return embeds

        def _get_mask(self, embeds: torch.Tensor, x: WavCondition) -> torch.Tensor:
            inputs = x.wav
            if self.interpolate_fn is not None and self.interpolate_stride == 0:
                # We do not mask interpolated embeddings, the outputs is [B, C].
                return torch.ones(x.wav.shape[:2], device=self.device, dtype=torch.bool)
            interpolate_downsampling_factor = self.interpolate_stride or 1
            downsampling_factor = x.sample_rate * interpolate_downsampling_factor / self._get_frame_rate()
            assert downsampling_factor.is_integer(), f"Downsampling factor {downsampling_factor} is not an integer."
            wav_mask = einops.rearrange(inputs, "b c t -> b 1 (c t)") != 0.0
            emb_mask = F.max_pool1d(
                wav_mask.float(), kernel_size=int(downsampling_factor), stride=int(downsampling_factor), padding=0
            )
            emb_mask = emb_mask.squeeze(1)  # [B, 1, C * S] -> [B, C * S]
            assert (
                emb_mask.shape[1] == embeds.shape[1]
            ), f"Mask shape {emb_mask.shape} does not match embedding shape {embeds.shape}."
            return emb_mask.bool()

    return MultiSpeakerWrapper


MultiSpeakerEncodecConditioner = wrap_conditioner_for_multi_speaker(EncodecConditioner)
MultiSpeakerWavLMConditioner = wrap_conditioner_for_multi_speaker(WavLMConditioner)

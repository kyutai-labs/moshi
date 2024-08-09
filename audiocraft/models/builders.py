# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import numpy as np
import typing as tp
import warnings

import audiocraft
import omegaconf
import torch
from torch import nn

from .encodec import (
    CompressionModel,
    EncodecModel,
    InterleaveStereoCompressionModel,
    EOSCompressionModel,
    MultistreamCompressionModel,
)
from .lm import LMModel

from ..conditioners import ConditionFuser, ConditionProvider, BaseConditioner
from ..modules.masking import ConstantSpanMask, GilbertMask
from ..modules.transformer import ProjectedTransformer
from .unet import DiffusionUnet
from .. import quantization as qt
from ..utils.utils import dict_from_config
from ..modules.diffusion_schedule import MultiBandProcessor, SampleProcessor


def get_quantizer(
        quantizer: str, cfg: omegaconf.DictConfig, encoder_dimension: int,
        input_dimension: int, output_dimension: int) -> qt.BaseQuantizer:
    kwargs = dict_from_config(getattr(cfg, quantizer))
    is_split_quantizer = quantizer == 'rvq' and kwargs.get('split')
    klass = {
        'no_quant': qt.DummyQuantizer,
        'rvq': qt.SplitResidualVectorQuantizer if is_split_quantizer else qt.ResidualVectorQuantizer,
    }[quantizer]
    kwargs.pop('split', None)
    if not is_split_quantizer:
        kwargs.pop('n_q_semantic', None)
        kwargs.pop('no_quantization_mode', None)
    kwargs.pop('detach_0', None)  # outdated param.
    if not kwargs.get('dimension'):
        # Compat with old models, we use the encoder dimension in that case.
        warnings.warn(
            "`rvq.dimension` was NOT provided. Will fallback on using "
            "the encoder dimension as a default, but this is deprecated.")
        kwargs['dimension'] = encoder_dimension
    kwargs['input_dimension'] = input_dimension
    kwargs['output_dimension'] = output_dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig, mask_fn: tp.Optional[nn.Module],
                            mask_position: tp.Optional[int]) -> tp.Tuple[nn.Module, nn.Module]:
    if encoder_name == 'seanet':
        kwargs = dict_from_config(getattr(cfg, 'seanet'))
        encoder_override_kwargs = kwargs.pop('encoder')
        decoder_override_kwargs = kwargs.pop('decoder')
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = audiocraft.modules.SEANetEncoder(**encoder_kwargs, mask_fn=mask_fn, mask_position=mask_position)
        decoder = audiocraft.modules.SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_encodec_transformers(cfg: omegaconf.DictConfig, encoder_dimension: int,
                             mae_pre: bool = False, mae_post: bool = False) -> tp.Tuple[tp.Optional[nn.Module],
                                                                                        tp.Optional[nn.Module]]:
    common_kwargs = dict_from_config(cfg.transformer)
    specialized_names = ['encoder', 'decoder']
    specialized_kwargs = [common_kwargs.pop(name) for name in specialized_names]
    maes = [mae_pre, mae_post]

    transformers: tp.List[tp.Optional[ProjectedTransformer]] = []
    mae_dimension = None
    encoder_transformer_output_dimension = encoder_dimension
    for name, kwargs in zip(specialized_names, specialized_kwargs):
        kwargs = {**common_kwargs, **kwargs}
        mae = maes.pop(0)
        if not kwargs.pop('use'):
            assert not mae, "MAE requires to use a Transformer in EnCodec."
            transformers.append(None)
            continue

        d_model = kwargs.pop('d_model')
        if d_model is None:
            d_model = encoder_dimension

        hidden_scale = kwargs.pop('hidden_scale')
        kwargs['dim_feedforward'] = int(d_model * hidden_scale)

        output_dimension = kwargs.pop('output_dimension', None)
        input_dimension = kwargs.pop('input_dimension', None)
        if name == 'encoder':
            if output_dimension is None:
                output_dimension = d_model
            if input_dimension is None:
                input_dimension = encoder_dimension
            encoder_transformer_output_dimension = output_dimension
        else:
            assert name == 'decoder'
            if output_dimension is None:
                output_dimension = encoder_dimension
            if input_dimension is None:
                input_dimension = encoder_transformer_output_dimension

        if mae_dimension is None:
            # Trying to support all possible combinations of mask_position
            # with having or not Transformers in the encoder / decoder...
            if cfg.encodec.mask_position == 'after_encoder':
                mae_dimension = encoder_dimension
            elif cfg.encodec.mask_position == 'after_quantizer':
                if name == 'encoder':
                    mae_dimension = output_dimension
                else:
                    assert name == 'decoder'
                    # If we got here, it means there is no Transformer in the encoder.
                    mae_dimension = encoder_dimension
            else:
                assert not mae
        output_dimensions = (output_dimension, mae_dimension) if mae else (output_dimension, )
        transformer = ProjectedTransformer(input_dimension, output_dimensions, d_model,
                                           device=cfg.device, conv_layout=True, **kwargs)
        transformers.append(transformer)
    assert len(transformers) == 2
    return transformers[0], transformers[1]


def get_mask_fn(cfg: omegaconf.DictConfig, mask_type: str, mask_framerate: int,
                dimension: int, return_mask: bool = True) -> tp.Optional[nn.Module]:
    if mask_type is None:
        return None
    elif mask_type == 'gilbert':
        kwargs = dict_from_config(getattr(cfg, 'gilbert'))
        masking_rate = kwargs.pop('masking_rate')
        mask_length = kwargs.pop('expected_burst_duration') * mask_framerate
        if 'init_std' not in kwargs:
            # When loading existing encodec checkpoints inside the LM Solver, this param doesn't exists.
            kwargs['init_std'] = 0.
        return GilbertMask(masking_rate, mask_length, conv_layout=True, dim=dimension,
                           dtype=getattr(torch, cfg.dtype), device=cfg.device, return_mask=return_mask, **kwargs)
    elif mask_type == 'constant_span':
        kwargs = dict_from_config(getattr(cfg, 'constant_span'))
        masking_rate = kwargs.pop('masking_rate')
        mask_length = kwargs.pop('mask_duration') * mask_framerate
        return ConstantSpanMask(masking_rate, mask_length, conv_layout=True, dim=dimension,
                                dtype=getattr(torch, cfg.dtype), device=cfg.device, return_mask=return_mask, **kwargs)
    else:
        raise KeyError(f"Unexpected mask_type {mask_type}")


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    """Instantiate a compression model."""
    if cfg.compression_model == 'encodec':
        kwargs = dict_from_config(getattr(cfg, 'encodec'))
        encoder_name = kwargs.pop('autoencoder')
        quantizer_name = kwargs.pop('quantizer')
        mae_pre = cfg.loss_weights.mae_pre > 0.
        mae_post = cfg.loss_weights.mae_post > 0.
        distill_wavlm = cfg.loss_weights.distill > 0.
        mask_position = kwargs.pop('mask_position')
        mask_type = kwargs.pop('mask_type')
        seanet_mask_fn = None
        seanet_mask_position = None
        if mask_position.startswith('after_conv'):
            ratios = list(reversed(cfg.seanet.ratios))
            position = int(mask_position.split('_')[-1])
            mask_framerate = cfg.sample_rate / float(np.prod(ratios[:position]))
            mask_dimension = cfg.seanet.n_filters * 2 ** position
            seanet_mask_fn = get_mask_fn(cfg, mask_type, mask_framerate=mask_framerate, dimension=mask_dimension,
                                         return_mask=False)
            seanet_mask_position = position
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg, seanet_mask_fn, seanet_mask_position)
        encoder_transformer, decoder_transformer = get_encodec_transformers(
            cfg, encoder.dimension, mae_pre, mae_post)
        # dimensions in/out of the RVQ depending on whether we have Transformers or not.
        input_dimension = encoder.dimension
        output_dimension = encoder.dimension
        if encoder_transformer is not None:
            input_dimension = encoder_transformer.output_dimensions[0]
        if decoder_transformer is not None:
            output_dimension = decoder_transformer.input_dimension
        quantizer = get_quantizer(
            quantizer_name, cfg, encoder.dimension, input_dimension, output_dimension)
        encoder_frame_rate = kwargs['sample_rate'] / encoder.hop_length
        frame_rate = kwargs.pop('frame_rate', None) or encoder_frame_rate
        renormalize = kwargs.pop('renormalize', False)
        outer_mask_fn = None
        if mask_position in ['after_quantizer', 'after_encoder']:
            outer_mask_fn = get_mask_fn(cfg, mask_type,
                                        mask_framerate=encoder_frame_rate, dimension=encoder.dimension)
        # deprecated params
        kwargs.pop('renorm', None)
        kwargs.pop('transformer_autocast', None)
        return EncodecModel(encoder, decoder, quantizer,
                            frame_rate=frame_rate, encoder_frame_rate=encoder_frame_rate, renormalize=renormalize,
                            encoder_transformer=encoder_transformer, decoder_transformer=decoder_transformer,
                            mask_fn=outer_mask_fn, mask_position=mask_position,
                            mae_pre=mae_pre, mae_post=mae_post, distill_wavlm=distill_wavlm, **kwargs).to(cfg.device)
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_lm_model(cfg: omegaconf.DictConfig, cardinality: tp.Optional[int] = None,
                 n_q: tp.Optional[int] = None) -> LMModel:
    """Instantiate a transformer LM."""
    if cfg.lm_model == 'transformer_lm':
        kwargs = dict_from_config(getattr(cfg, 'transformer_lm'))
        kwargs.pop('special_mode')  # compat with experimental models.
        if 'has_control' in kwargs:
            # compat with first round of interleaved model.
            has_control = kwargs.pop('has_control')
            assert not has_control, 'no longer backward compat'
        cfg_n_q = kwargs['n_q']
        cfg_cardinality = kwargs['card']
        if cfg_n_q is None:
            assert n_q is not None
            kwargs['n_q'] = n_q
        else:
            if n_q is not None:
                assert cfg_n_q == n_q, f"Mismatch of n_q {cfg_n_q} vs. {n_q}"
            n_q = cfg_n_q
        if cfg_cardinality is None:
            kwargs['card'] = cardinality
        elif cardinality is not None:
            assert cfg_cardinality == cardinality, f"Mismatch of n_q {cfg_cardinality} vs. {cardinality}"

        assert n_q is not None  # making type happy.

        fuser = get_condition_fuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
        if len(fuser.fuse2cond['cross']) > 0:  # enforce cross-att programmatically
            kwargs['cross_attention'] = True
        delays: tp.Optional[tp.List[int]] = kwargs.pop('delays', None)
        if delays is None:
            if cfg.codebooks_pattern.modeling == 'parallel':
                delays = [0]
            elif cfg.codebooks_pattern.modeling == 'delay':
                delays = cfg.codebooks_pattern.delay.delays
            else:
                raise RuntimeError("Impossible to convert codebook pattern to delay only.")
            assert delays is not None

        return LMModel(
            condition_provider=condition_provider,
            fuser=fuser,
            delays=delays,
            cfg_dropout=cfg.classifier_free_guidance.training_dropout,
            cfg_coef=cfg.classifier_free_guidance.inference_coef,
            attribute_dropouts=dict(cfg.attribute_dropouts),
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            **kwargs
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected LM model {cfg.lm_model}")


def get_conditioner(output_dim: int, device: str,
                    conditioner_cfg: omegaconf.DictConfig) -> BaseConditioner:
    conditioner_type = conditioner_cfg.type
    conditioner_kwargs = dict_from_config(conditioner_cfg[conditioner_type])
    conditioner_kwargs.update({'output_dim': output_dim, 'device': device})
    # All imports are lazy to avoid slow startup times when not used.
    if conditioner_type == 't5':
        from ..conditioners.text import T5Conditioner
        return T5Conditioner(**conditioner_kwargs)
    if conditioner_type == 'mt5':
        from ..conditioners.text import MultilingualT5Conditioner
        return MultilingualT5Conditioner(**conditioner_kwargs)
    elif conditioner_type == 'lut':
        from ..conditioners.text import LUTConditioner
        return LUTConditioner(**conditioner_kwargs)
    elif conditioner_type == 'noise_level':
        from ..conditioners.audio import NoiseLevelConditioner
        return NoiseLevelConditioner(**conditioner_kwargs)
    elif conditioner_type == 'encodec':
        from ..conditioners.audio import EncodecConditioner
        return EncodecConditioner(**conditioner_kwargs)
    elif conditioner_type == 'wavlm':
        from ..conditioners.audio import WavLMConditioner
        return WavLMConditioner(**conditioner_kwargs)
    elif conditioner_type == 'encodec_multi_speaker':
        from ..conditioners.audio import MultiSpeakerEncodecConditioner
        return MultiSpeakerEncodecConditioner(**conditioner_kwargs)
    elif conditioner_type == 'wavlm_multi_speaker':
        from ..conditioners.audio import MultiSpeakerWavLMConditioner
        return MultiSpeakerWavLMConditioner(**conditioner_kwargs)
    else:
        raise RuntimeError(f"Unknow conditioner type {conditioner_type}.")


def get_conditioner_provider(output_dim: int, cfg: omegaconf.DictConfig) -> ConditionProvider:
    """Instantiate a conditioning model."""
    device = cfg.device
    conditioners: tp.Dict[str, BaseConditioner] = {}
    for cond, cond_cfg in cfg.conditioners.items():
        conditioners[cond] = get_conditioner(output_dim, device, cond_cfg)
    conditioner = ConditionProvider(conditioners, device=device)
    return conditioner


def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = dict_from_config(getattr(cfg, 'fuser'))
    fuser_methods = ['sum', 'cross', 'prepend']
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_debug_compression_model(device='cpu', sample_rate: int = 32000):
    """Instantiate a debug compression model to be used for unit tests."""
    assert sample_rate in [16000, 32000], "unsupported sample rate for debug compression model"
    model_ratios = {
        16000: [10, 8, 8],  # 25 Hz at 16kHz
        32000: [10, 8, 16]  # 25 Hz at 32kHz
    }
    ratios: tp.List[int] = model_ratios[sample_rate]
    frame_rate = 25
    seanet_kwargs: dict = {
        'n_filters': 4,
        'n_residual_layers': 1,
        'dimension': 32,
        'ratios': ratios,
    }
    encoder = audiocraft.modules.SEANetEncoder(**seanet_kwargs)
    decoder = audiocraft.modules.SEANetDecoder(**seanet_kwargs)
    quantizer = qt.ResidualVectorQuantizer(dimension=32, bins=400, n_q=4)
    init_x = torch.randn(8, 32, 128)
    quantizer(init_x, 1)  # initialize kmeans etc.
    compression_model = EncodecModel(
        encoder, decoder, quantizer,
        frame_rate=frame_rate, encoder_frame_rate=frame_rate,
        sample_rate=sample_rate, channels=1).to(device)
    return compression_model.eval()


def get_diffusion_model(cfg: omegaconf.DictConfig):
    # TODO Find a way to infer the channels from dset
    channels = cfg.channels
    num_steps = cfg.schedule.num_steps
    return DiffusionUnet(
            chin=channels, num_steps=num_steps, **cfg.diffusion_unet)


def get_processor(cfg, sample_rate: int = 24000):
    sample_processor = SampleProcessor()
    if cfg.use:
        kw = dict(cfg)
        kw.pop('use')
        kw.pop('name')
        if cfg.name == "multi_band_processor":
            sample_processor = MultiBandProcessor(sample_rate=sample_rate, **kw)
    return sample_processor


def get_debug_lm_model(device='cpu'):
    """Instantiate a debug LM to be used for unit tests."""
    dim = 16
    from audiocraft.conditioners.text import LUTConditioner
    providers = {
        'description': LUTConditioner(n_bins=128, dim=dim, output_dim=dim, tokenizer="whitespace", device="cpu"),
    }
    condition_provider = ConditionProvider(providers)
    fuser = ConditionFuser({'cross': ['description'], 'prepend': [], 'sum': [], })
    lm = LMModel(
        condition_provider, fuser, delays=[0],
        n_q=4, card=400, dim=dim, num_heads=4, num_layers=2,
        cross_attention=True, causal=True)
    return lm.to(device).eval()


def get_wrapped_compression_model(
        compression_model: CompressionModel,
        cfg: omegaconf.DictConfig) -> CompressionModel:
    if hasattr(cfg, 'interleave_stereo_codebooks'):
        if cfg.interleave_stereo_codebooks.use:
            kwargs = dict_from_config(cfg.interleave_stereo_codebooks)
            kwargs.pop('use')
            compression_model = InterleaveStereoCompressionModel(compression_model, **kwargs)
    if hasattr(cfg, 'compression_model_n_q'):
        if cfg.compression_model_n_q is not None:
            compression_model.set_num_codebooks(cfg.compression_model_n_q)
    if hasattr(cfg.tokens, 'use_eos'):
        if cfg.tokens.use_eos:
            assert cfg.tokens.padding_with_zero_token, "EOS token requires padding with zero token."
            compression_model = EOSCompressionModel(compression_model)
    if hasattr(cfg.tokens, 'multistream'):
        if cfg.tokens.multistream:
            kwargs = dict_from_config(cfg.multistream)
            kwargs.pop('loss_only_on_main')
            kwargs.pop('loss_scale_on_other')
            compression_model = MultistreamCompressionModel(compression_model, num_sources=2, **kwargs)
    return compression_model

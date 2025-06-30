# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub.errors import EntryNotFoundError
except ImportError:
    from huggingface_hub.utils import EntryNotFoundError  # pyright: ignore
from safetensors.torch import load_model, load_file
import sentencepiece
import torch
import typing as tp
from .compression import MimiModel
from ..conditioners import BaseConditioner, ConditionProvider, ConditionFuser
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer
from ..modules.lora import replace_all_linear_with_lora, replace_lora_with_linear


SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
MOSHI_NAME = "model.safetensors"
MOSHI_Q8_NAME = "model.q8.safetensors"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def hf_get(filename: str | Path, hf_repo: str | None = None,
           check_local_file_exists: bool = False) -> Path:
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename.removeprefix("hf://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    elif hf_repo is not None:
        if check_local_file_exists:
            if Path(filename).exists():
                return Path(filename)
        return Path(hf_hub_download(hf_repo, filename))
    else:
        return Path(filename)


@dataclass
class CheckpointInfo:
    """
    Contains the paths to each sub model, along with some extra configuration.

    Args:
        moshi_weights: path to the checkpoint for the Moshi LM.
        mimi_weights: path to the checkpoint for the Mimi audio tokenizer.
        tokenizer: path to the text tokenizer.
        lm_config: config for instantiating the LM model.
            Can be None if the original Moshi 7B config should be used.
        raw_config: raw config, including original keys not intended for the LM.
        model_type: indicate the intended use, should be `moshi` or `hibiki`.
        lora_weights: path to an optional checkpoint with lora weights.
        lm_gen_config: optional default params to use for generation with this model.
        tts_config: optional TTS specific configuration.
        stt_config: optional STT specific configuration.
        model_id: optional dict containing tracability information on the model origin, in particular
            its signature and epoch.
    """

    moshi_weights: Path
    mimi_weights: Path
    tokenizer: Path
    lm_config: dict | None = None
    raw_config: dict | None = None
    model_type: str = "moshi"
    lora_weights: Path | None = None
    lm_gen_config: dict = field(default_factory=dict)
    tts_config: dict = field(default_factory=dict)
    stt_config: dict = field(default_factory=dict)
    model_id: dict = field(default_factory=dict)

    @staticmethod
    def from_hf_repo(
        hf_repo: str,
        moshi_weights: Path | str | None = None,
        mimi_weights: Path | str | None = None,
        tokenizer: Path | str | None = None,
        config_path: Path | str | None = None,
        lora_weights: Path | str | None = None,
    ) -> "CheckpointInfo":
        """Downloads the checkpoints from the given repo, along with its config.

        Extra overrides are possible for each of Moshi, Mimi, or the text tokenizer,
        which should be either a Path to a local file or a string representing a path
        to a local file or starting with `hf://` for pointing to a file in another repo.

        Finally, a `config_path` can be provided to override the config from the repository.
        """
        if config_path is None:
            try:
                config_path = hf_hub_download(hf_repo, "config.json")
            except EntryNotFoundError:
                # No config.json, which might indicate legacy repository.
                warnings.warn(
                    f"Repository {hf_repo} contains no config.json. "
                    "Assuming this is a Moshi 7B. Support for such repository "
                    "might be removed in the future."
                )
        if config_path is None:
            moshi_name = MOSHI_NAME
            mimi_name = MIMI_NAME
            tokenizer_name = TEXT_TOKENIZER_NAME
            lm_config = None
            raw_config = None
            model_type = "moshi"
            lm_gen_config = {}
            tts_config = {}
            stt_config = {}
            model_id = {}
            lora_name = None
        else:
            raw_config = json.loads(Path(config_path).read_text())
            lm_config = dict(raw_config)
            moshi_name = lm_config.pop("moshi_name", MOSHI_NAME)
            mimi_name = lm_config.pop("mimi_name", MIMI_NAME)
            tokenizer_name = lm_config.pop("tokenizer_name", TEXT_TOKENIZER_NAME)
            lora_name = lm_config.pop("lora_name", None)
            model_type = lm_config.pop("model_type", "moshi")
            lm_gen_config = lm_config.pop("lm_gen_config", {})
            tts_config = lm_config.pop("tts_config", {})
            stt_config = lm_config.pop("stt_config", {})
            model_id = lm_config.pop("model_id", {})

        if moshi_weights is None:
            moshi_weights_final = hf_get(moshi_name, hf_repo)
        else:
            moshi_weights_final = hf_get(moshi_weights)

        if mimi_weights is None:
            mimi_weights_final = hf_get(mimi_name, hf_repo)
        else:
            mimi_weights_final = hf_get(mimi_weights)

        if tokenizer is None:
            tokenizer_final = hf_get(tokenizer_name, hf_repo)
        else:
            tokenizer_final = hf_get(tokenizer)

        if lora_weights is None and lora_name:
            lora_weights_final = hf_get(lora_name, hf_repo)
        elif lora_weights is not None:
            lora_weights_final = hf_get(lora_weights)
        else:
            lora_weights_final = None

        return CheckpointInfo(
            moshi_weights_final,
            mimi_weights_final,
            tokenizer_final,
            lm_config,
            raw_config,
            model_type,
            lora_weights_final,
            lm_gen_config=lm_gen_config,
            tts_config=tts_config,
            stt_config=stt_config,
            model_id=model_id,
        )

    def get_mimi(self, device: torch.device | str = "cpu") -> MimiModel:
        if self.lm_config is None:
            num_codebooks = 8
        else:
            num_codebooks = max(self.lm_config["dep_q"], self.lm_config["n_q"] - self.lm_config["dep_q"])
        return get_mimi(self.mimi_weights, num_codebooks=num_codebooks, device=device)

    def get_moshi(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        load_weight: bool = True,
        **kwargs,
    ) -> LMModel:
        model = get_moshi_lm(
            self.moshi_weights if load_weight else None,
            lm_kwargs=self.lm_config,
            device=device,
            dtype=dtype,
            lora_weights=self.lora_weights,
            **kwargs,
        )
        if self.model_type == "hibiki":
            # Sometime the model samples the EOS (2) too early, which we want to ignore.
            # We keep generating if the input file is not finished, and this is a way
            # to implicitely replace early EOS with PAD.
            model.text_emb.weight.data[2] = model.text_emb.weight.data[3]
        return model

    def get_text_tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        return sentencepiece.SentencePieceProcessor(str(self.tokenizer))  # type: ignore


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(
    filename: str | Path | None, device: torch.device | str = "cpu", num_codebooks: int = 8
) -> MimiModel:
    """Return a pretrained Mimi model, or unintialized if `filename` is None."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if filename is not None:
        if _is_safetensors(filename):
            load_model(model, filename, device=str(device))
        else:
            pkg = torch.load(filename, "cpu")
            model.load_state_dict(pkg["model"])
    model.set_num_codebooks(num_codebooks)
    return model


def get_moshi_lm(
    filename: str | Path | None,
    lm_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_weights: str | Path | None = None,
    fuse_lora: bool = False,
    lm_kwargs_overrides={},
) -> LMModel:
    if lm_kwargs is None:
        lm_kwargs = _lm_kwargs
    lm_kwargs = dict(lm_kwargs)
    assert lm_kwargs is not None

    if "conditioners" in lm_kwargs:
        lm_kwargs["condition_provider"] = get_conditioner_provider(
            lm_kwargs["dim"], device, lm_kwargs
        )
        del lm_kwargs["conditioners"]
    if "fuser" in lm_kwargs:
        lm_kwargs["fuser"] = get_condition_fuser(lm_kwargs)

    lm_kwargs = lm_kwargs | lm_kwargs_overrides
    assert lm_kwargs is not None

    # deprecated params.
    lm_kwargs.pop("depformer_causal", None)

    # moved params
    if 'demux_second_stream' in lm_kwargs:
        lm_kwargs['demux_second_text_stream'] = lm_kwargs.pop('demux_second_stream')

    # lora params.
    lora = lm_kwargs.pop("lora", False)
    lora_rank = lm_kwargs.pop("lora_rank", 128)
    lora_scaling = lm_kwargs.pop("lora_scaling", 2.0)

    init_device = device
    if filename is not None:
        init_device = torch.device('meta')

    model = LMModel(
        device=init_device,
        dtype=dtype,
        **lm_kwargs)

    if filename is not None:
        if _is_safetensors(filename):
            state = load_file(filename, device=str(device))
            for key, value in state.items():
                if value.dtype.is_floating_point:
                    if key.startswith('condition_provider.') or key.startswith('fuser.'):
                        value = value.float()
                    else:
                        value = value.to(dtype)
                state[key] = value
            model.load_state_dict(state, assign=True)

        else:
            pkg = torch.load(filename, "cpu",)
            model.load_state_dict(pkg["fsdp_best_state"]["model"], assign=True)

    if lora:
        assert not lm_kwargs.get("quantize"), (
            "LoRA and quantization are incompatible for now."
        )
        model = get_lora_moshi(
            model=model,
            lora_rank=lora_rank,
            lora_scaling=lora_scaling,
            lora_weights=lora_weights,
            device=device,
            dtype=dtype,
            fuse_lora=fuse_lora,
        )
    else:
        assert lora_weights is None, (
            "`lora` is False, but received some lora_weights to load."
        )
    model.eval()
    return model


def get_conditioner(
    output_dim: int, device: torch.device | str, conditioner_cfg: dict
) -> BaseConditioner:
    conditioner_type = conditioner_cfg["type"]
    conditioner_kwargs = conditioner_cfg[conditioner_type]
    conditioner_kwargs.update({"output_dim": output_dim, "device": device})
    if conditioner_type == "lut":
        from ..conditioners.text import LUTConditioner
        return LUTConditioner(**conditioner_kwargs)
    elif conditioner_type == "tensor":
        from ..conditioners.tensors import TensorConditioner
        return TensorConditioner(**conditioner_kwargs)
    else:
        raise RuntimeError(f"Unknow conditioner type {conditioner_type}.")


def get_conditioner_provider(
    output_dim: int, device: torch.device | str, cfg: dict
) -> ConditionProvider:
    """Instantiate a conditioning model."""
    conditioners: tp.Dict[str, BaseConditioner] = {}
    for cond, cond_cfg in cfg["conditioners"].items():
        conditioners[cond] = get_conditioner(output_dim, device, cond_cfg)
    conditioner = ConditionProvider(conditioners, device=device)
    return conditioner


def get_condition_fuser(cfg: dict) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = cfg["fuser"]
    fuser_methods = ["sum", "cross", "prepend"]
    fuse2cond = {k: fuser_cfg.get(k, []) for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_lora_moshi(
    model: LMModel,
    lora_weights: str | Path | None,
    lora_rank: int,
    lora_scaling: float,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    fuse_lora: bool = True,
) -> LMModel:
    init_device = device
    if lora_weights is not None:
        init_device = torch.device('meta')
    replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=init_device)
    if lora_weights is not None:
        assert _is_safetensors(lora_weights), "LoRA weights must be a safetensors file."
        lora_state_dict = load_file(lora_weights, device=str(device))
        for key, value in lora_state_dict.items():
            if value.dtype.is_floating_point:
                value = value.to(dtype=dtype)
            lora_state_dict[key] = value
        res = model.load_state_dict(lora_state_dict, strict=False, assign=True)
        if res.unexpected_keys:
            raise RuntimeError(
                f"unexpected_keys in the lora weights: {res.unexpected_keys}"
            )
        model = model.to(dtype=dtype, device=device)
        if fuse_lora:
            replace_lora_with_linear(model)
    return model

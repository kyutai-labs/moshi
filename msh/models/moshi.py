import msh
import torch
import safetensors
import numpy as np
import random
from pathlib import Path
import typing as tp

SAMPLE_RATE = 24000
FRAME_RATE = 12.5


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(42424242)


seanet_kwargs = {
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
    "lstm": 0,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": seanet_kwargs["dimension"],
    "output_dimension": seanet_kwargs["dimension"],
}
transformer_kwargs = {
    "d_model": seanet_kwargs["dimension"],
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
    "input_dimension": seanet_kwargs["dimension"],
    "output_dimensions": [seanet_kwargs["dimension"]],
}

lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "card": quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "real_rms_norm_f32",
    "positional_embedding": "rope",
    "depformer": bool,
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
}


def _is_safetensors(filename: tp.Union[str, Path]) -> bool:
    filename = Path(filename)
    return filename.suffix in (".safetensors", ".sft", ".sfts")


def get_encodec(filename: tp.Union[str, Path], device):
    encoder = msh.modules.SEANetEncoder(**seanet_kwargs)
    decoder = msh.modules.SEANetDecoder(**seanet_kwargs)
    encoder_transformer = msh.modules.transformer.ProjectedTransformer(
        **transformer_kwargs
    )
    decoder_transformer = msh.modules.transformer.ProjectedTransformer(
        **transformer_kwargs
    )
    quantizer = msh.quantization.SplitResidualVectorQuantizer(
        **quantizer_kwargs,
    )
    model = msh.models.EncodecModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        renormalize=False,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        safetensors.torch.load_model(model, filename)
    else:
        pkg = torch.load(
            filename,
            "cpu",
        )
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def get_lm(filename: tp.Union[str, Path], device):
    model = msh.models.LMModel(
        delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        **lm_kwargs,
    ).to(device=device)
    model.eval()
    model.to(torch.bfloat16)
    if _is_safetensors(filename):
        safetensors.torch.load_model(model, filename)
    else:
        pkg = torch.load(
            filename,
            "cpu",
        )
        model.load_state_dict(pkg["fsdp_best_state"]["model"])
    model.autocast = msh.utils.autocast.TorchAutocast(
        enabled=True, dtype=torch.bfloat16, device_type="cuda"
    )
    return model

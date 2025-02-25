from pathlib import Path
from safetensors.torch import load_file, load_model
import torch

from moshi.models import lm
from moshi.utils.utils import cross_entropy


def _get_assets() -> Path:
    return Path(__file__).parent / 'assets'


def _get_lm(device=None, dtype=torch.float32) -> lm.LMModel:
    torch.manual_seed(1234)
    model = lm.LMModel(
        delays=[0, 1, 2, 4],
        n_q=3,
        dep_q=3,
        card=32,
        text_card=48,
        dim=16,
        num_layers=2,
        num_heads=1,
        hidden_scale=1,
        depformer_dim=16,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=[0, 1, 1],
        depformer_low_rank_embeddings=8,
        depformer_num_heads=1,
        depformer_gating='silu',
        context=4,
        device=device,
        dtype=dtype,
    )
    return model


def test_init():
    _get_lm(dtype=torch.float32)
    _get_lm(dtype=torch.bfloat16)
    _get_lm(dtype=torch.float16)


@torch.no_grad
def test_forward():
    model = _get_lm()
    load_model(model, _get_assets() / 'test_lm_model.safetensors')
    codes = load_file(_get_assets() / 'test_lm_codes.safetensors')['codes']
    out = model(codes)
    assert out.logits is not None
    assert out.text_logits is not None
    assert out.mask.shape == codes[:, 1:].shape
    assert out.text_mask.shape == codes[:, :1].shape
    assert out.logits.shape[:-1] == codes[:, 1:].shape
    assert out.logits.shape[-1] == model.card
    assert out.text_logits.shape[-1] == model.text_card

    ref_out = load_file(_get_assets() / 'test_lm_out.safetensors')
    assert (ref_out['mask'] == out.mask).all()
    assert (ref_out['text_mask'] == out.text_mask).all()
    ce = cross_entropy(out.logits, codes[:, 1:], out.mask)
    ce_ref = cross_entropy(ref_out['logits'], codes[:, 1:], out.mask)
    delta = (ce.mean(dim=(0, 2)) - ce_ref.mean(dim=(0, 2))).abs() / ce_ref.mean(dim=(0, 2))
    assert delta.amax() <= 1e-6, delta.amax()

    ce = cross_entropy(out.text_logits, codes[:, :1], out.text_mask)
    ce_ref = cross_entropy(ref_out['text_logits'], codes[:, :1], out.text_mask)
    delta = (ce.mean(dim=(0, 2)) - ce_ref.mean(dim=(0, 2))).abs() / ce_ref.mean(dim=(0, 2))
    assert delta.amax() <= 1e-6, delta.amax()

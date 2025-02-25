from moshi.models import lm
import torch


def _get_lm(device=None, dtype=torch.float32) -> lm.LMModel:
    torch.manual_seed(1234)
    model = lm.LMModel(
        delays=[0, 1, 2, 4],
        n_q=3,
        dep_q=3,
        card=32,
        text_card=48,
        dim=64,
        num_layers=2,
        num_heads=1,
        hidden_scale=1,
        depformer_dim=64,
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


def test_forward():
    model = _get_lm()

    codes = torch.randint(model.card, (3, model.num_codebooks, 7))
    out = model(codes)
    assert out.logits is not None
    assert out.text_logits is not None

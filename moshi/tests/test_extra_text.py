import torch

from moshi.models import lm


def _get_lm(extra_text_stream_depth=0, **kwargs) -> lm.LMModel:
    torch.manual_seed(1234)
    n_q = 3
    dep_q = 3
    num_text_streams = 1 + extra_text_stream_depth
    delays = [0] * num_text_streams + [1, 2, 4]
    model = lm.LMModel(
        delays=delays,
        n_q=n_q,
        dep_q=dep_q,
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
        depformer_gating="silu",
        context=4,
        extra_text_stream_depth=extra_text_stream_depth,
        **kwargs,
    )
    return model


def _run_streaming(model: lm.LMModel, lm_gen: lm.LMGen, num_steps: int = 10):
    n_input = model.num_audio_codebooks - model.dep_q
    tokens_out = []
    with lm_gen.streaming(1):
        for step in range(num_steps):
            input_tokens = torch.randint(0, model.card, (1, n_input, 1))
            out = lm_gen.step(input_tokens)
            if out is not None:
                tokens_out.append(out)
    return tokens_out


def test_model_properties():
    model_0 = _get_lm(extra_text_stream_depth=0)
    assert model_0.num_text_streams == 1
    assert model_0.num_codebooks == 4
    assert model_0.audio_offset == 1

    model_1 = _get_lm(extra_text_stream_depth=1)
    assert model_1.num_text_streams == 2
    assert model_1.num_codebooks == 5
    assert model_1.audio_offset == 2


def test_extra_text_linears():
    model_0 = _get_lm(extra_text_stream_depth=0)
    assert len(model_0.extra_text_linears) == 0
    assert len(model_0.proj_text_embs) == 0

    model_1 = _get_lm(extra_text_stream_depth=1)
    assert len(model_1.extra_text_linears) == 1
    assert len(model_1.proj_text_embs) == 2

    model_2 = _get_lm(extra_text_stream_depth=2)
    assert len(model_2.extra_text_linears) == 2
    assert len(model_2.proj_text_embs) == 3


def test_initial_token_shape():
    model_0 = _get_lm(extra_text_stream_depth=0)
    assert model_0._get_initial_token().shape == (1, 4, 1)

    model_1 = _get_lm(extra_text_stream_depth=1)
    assert model_1._get_initial_token().shape == (1, 5, 1)


@torch.no_grad()
def test_streaming_no_extra_text():
    model = _get_lm(extra_text_stream_depth=0)
    model.eval()
    lm_gen = lm.LMGen(model)
    tokens_out = _run_streaming(model, lm_gen)
    assert len(tokens_out) > 0
    for t in tokens_out:
        assert t.shape == (1, model.dep_q + 1, 1)


@torch.no_grad()
def test_streaming_extra_text_no_autoregressive():
    model = _get_lm(extra_text_stream_depth=1)
    model.eval()

    lm_gen = lm.LMGen(model, user_text_autoregressive=False)
    tokens_out = _run_streaming(model, lm_gen)
    assert len(tokens_out) > 0
    for t in tokens_out:
        assert t.shape == (1, model.num_text_streams + model.dep_q, 1)
        assert t[0, 1, 0].item() == model.zero_token_id


@torch.no_grad()
def test_streaming_extra_text_autoregressive():
    model = _get_lm(extra_text_stream_depth=1)
    model.eval()

    lm_gen = lm.LMGen(model, user_text_autoregressive=True)
    tokens_out = _run_streaming(model, lm_gen)
    assert len(tokens_out) > 0
    saw_real_token = False
    for t in tokens_out:
        assert t.shape == (1, model.num_text_streams + model.dep_q, 1)
        user_token = t[0, 1, 0].item()
        if user_token != model.zero_token_id:
            assert 0 <= user_token <= model.text_card
            saw_real_token = True
    assert saw_real_token


@torch.no_grad()
def test_autoregressive_changes_output():
    model = _get_lm(extra_text_stream_depth=1)
    model.eval()

    n_input = model.num_audio_codebooks - model.dep_q
    torch.manual_seed(99)
    inputs = [torch.randint(0, model.card, (1, n_input, 1)) for _ in range(12)]

    def run(autoregressive):
        torch.manual_seed(42)
        gen = lm.LMGen(model, user_text_autoregressive=autoregressive)
        results = []
        with gen.streaming(1):
            for inp in inputs:
                out = gen.step(inp)
                if out is not None:
                    results.append(out.clone())
        return results

    out_no_ar = run(False)
    out_ar = run(True)
    assert len(out_no_ar) == len(out_ar)
    any_diff = any(not torch.equal(a, b) for a, b in zip(out_no_ar, out_ar))
    assert any_diff, "Autoregressive feedback should change model outputs"


@torch.no_grad()
def test_forward_text_with_proj():
    model = _get_lm(extra_text_stream_depth=1)
    model.eval()
    B, S = 2, 3
    sequence = torch.randint(0, min(model.card, model.text_card), (B, model.num_codebooks, S))
    transformer_out, text_logits = model.forward_text(sequence)
    assert transformer_out.shape == (B, S, model.dim)
    assert text_logits.shape == (B, model.num_text_streams, S, model.text_card)


@torch.no_grad()
def test_forward_text_baseline_shape():
    model = _get_lm(extra_text_stream_depth=0)
    model.eval()
    B, S = 2, 3
    sequence = torch.randint(0, min(model.card, model.text_card), (B, model.num_codebooks, S))
    transformer_out, text_logits = model.forward_text(sequence)
    assert transformer_out.shape == (B, S, model.dim)
    assert text_logits.shape == (B, 1, S, model.text_card)

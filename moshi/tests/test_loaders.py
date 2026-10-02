import torch
from safetensors.torch import save_file

from moshi.models import loaders


class _RecordingLM:
    """Stand-in for LMModel that records the state dict handed to load_state_dict."""

    def __init__(self, **kwargs):
        self.state = None

    def load_state_dict(self, state, assign=False):
        self.state = state

    def eval(self):
        return self


def _load(tmp_path, monkeypatch, tensors, dtype=torch.bfloat16):
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    holder = {}

    def _make(**kwargs):
        holder["model"] = _RecordingLM(**kwargs)
        return holder["model"]

    monkeypatch.setattr(loaders, "LMModel", _make)
    loaders.get_moshi_lm(path, lm_kwargs={}, device="cpu", dtype=dtype)
    return holder["model"].state


def test_float_weights_are_cast_to_requested_dtype(tmp_path, monkeypatch):
    state = _load(tmp_path, monkeypatch, {"linear.weight": torch.zeros(2, 2)})
    assert state["linear.weight"].dtype == torch.bfloat16


def test_conditioner_weights_stay_float32(tmp_path, monkeypatch):
    state = _load(
        tmp_path,
        monkeypatch,
        {"condition_provider.w": torch.zeros(2), "fuser.w": torch.zeros(2)},
    )
    assert state["condition_provider.w"].dtype == torch.float32
    assert state["fuser.w"].dtype == torch.float32


def test_int8_scale_buffers_stay_float32(tmp_path, monkeypatch):
    """Quantized (q8) checkpoints store bitsandbytes scales as float32 tensors.

    They come in a dotted form (`...weight_scb`) and an undotted form for the
    fused attention parameter (`...in_proj_weight_scb`); both must keep their
    dtype, otherwise generation fails with "Expected weight_scb to have type
    float, but got bfloat16".
    """
    state = _load(
        tmp_path,
        monkeypatch,
        {
            "transformer.layers.0.gating.linear_in.weight_scb": torch.ones(4),
            "transformer.layers.0.self_attn.in_proj_weight_scb": torch.ones(4),
            "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(4, 4),
        },
    )
    assert state["transformer.layers.0.gating.linear_in.weight_scb"].dtype == torch.float32
    assert state["transformer.layers.0.self_attn.in_proj_weight_scb"].dtype == torch.float32
    assert state["transformer.layers.0.self_attn.in_proj_weight"].dtype == torch.bfloat16

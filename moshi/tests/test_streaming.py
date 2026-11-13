import gc
import weakref

import pytest
import torch
from torch import nn

from moshi.models import lm
from moshi.modules.streaming import StreamingContainer


@pytest.fixture
def gc_disabled():
    # With the cyclic collector off, an object is freed only by reference counting,
    # so anything still alive after `del` is held by a reference cycle.
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _container() -> StreamingContainer:
    module = StreamingContainer()
    module.lin = nn.Linear(2, 2)
    return module


def _tree() -> StreamingContainer:
    root = _container()
    root.a = _container()
    root.a.b = _container()
    root.seq = nn.Sequential(nn.Linear(2, 2), _container())
    root.det = _container()
    root.det.set_streaming_detached(True)
    root.det.c = _container()
    root.z = _container()
    return root


def test_streamed_module_freed_without_gc(gc_disabled):
    module = _tree()
    with module.streaming(2):
        module.reset_streaming()
        module.set_exec_mask(torch.tensor([True, False]))
        module.get_streaming_state()
    ref = weakref.ref(module)
    del module
    assert ref() is None, "module outlived its last reference"


@torch.no_grad()
def test_lm_gen_freed_without_gc(gc_disabled):
    torch.manual_seed(1234)
    model = lm.LMModel(
        delays=[0, 1, 2, 4], n_q=3, dep_q=3, card=32, text_card=48, dim=16, num_layers=2,
        num_heads=1, hidden_scale=1, depformer_dim=16, depformer_multi_linear=True,
        depformer_weights_per_step=True, depformer_weights_per_step_schedule=[0, 1, 1],
        depformer_low_rank_embeddings=8, depformer_num_heads=1, depformer_gating='silu',
        context=4).eval()
    lm_gen = lm.LMGen(model)
    with lm_gen.streaming(1):
        for _ in range(6):  # Enough steps to get past the delays and produce tokens.
            lm_gen.step(torch.zeros(1, model.num_audio_codebooks, 1, dtype=torch.long))
    ref = weakref.ref(lm_gen)
    del lm_gen
    assert ref() is None, "LMGen outlived its last reference"


def test_streaming_traversal_order():
    root = _tree()
    expected = ["", "a", "a.b", "seq.1", "z"]
    with root.streaming(1):
        for _ in range(2):  # The second call uses the cached children.
            state = root.get_streaming_state()
            assert list(state) == expected
            for name in expected:
                assert state[name] is root.get_submodule(name)._streaming_state
        # A detached child, and its children, only stream when asked directly.
        assert not root.det.is_streaming
        assert not root.det.c.is_streaming
    assert not any(root.get_submodule(name).is_streaming for name in expected)


def test_detached_module_streams_when_asked_directly():
    root = _tree()
    with root.det.streaming(1):
        assert list(root.det.get_streaming_state()) == ["", "c"]
        assert root.det.is_streaming and root.det.c.is_streaming
        assert not root.is_streaming
    assert not root.det.is_streaming and not root.det.c.is_streaming

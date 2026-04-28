# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for StreamingAdd.reset_state() and
SeanetResnetBlock.reset_state() — verifies that stale partial-frame
buffers do not leak across successive streaming generations.

See: https://github.com/kyutai-labs/moshi/issues/407
"""

import pytest

mlx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

from moshi_mlx.modules.seanet import StreamingAdd, SeanetConfig, SeanetResnetBlock


# ---------------------------------------------------------------------------
# StreamingAdd unit tests
# ---------------------------------------------------------------------------


class TestStreamingAddReset:
    """StreamingAdd.reset_state() must clear _lhs and _rhs."""

    def test_reset_clears_lhs(self):
        sa = StreamingAdd()
        # Produce a leftover in _lhs by feeding unequal lengths (lhs longer).
        lhs = mlx.ones((1, 1, 5))
        rhs = mlx.ones((1, 1, 3))
        sa.step(lhs, rhs)
        assert sa._lhs is not None, "_lhs should hold leftover samples"

        sa.reset_state()
        assert sa._lhs is None, "_lhs must be None after reset"
        assert sa._rhs is None, "_rhs must be None after reset"

    def test_reset_clears_rhs(self):
        sa = StreamingAdd()
        # Produce a leftover in _rhs by feeding unequal lengths (rhs longer).
        lhs = mlx.ones((1, 1, 3))
        rhs = mlx.ones((1, 1, 5))
        sa.step(lhs, rhs)
        assert sa._rhs is not None, "_rhs should hold leftover samples"

        sa.reset_state()
        assert sa._lhs is None
        assert sa._rhs is None

    def test_equal_length_no_residual(self):
        sa = StreamingAdd()
        lhs = mlx.ones((1, 1, 4))
        rhs = mlx.ones((1, 1, 4))
        out = sa.step(lhs, rhs)
        assert sa._lhs is None
        assert sa._rhs is None
        assert out.shape[-1] == 4

    def test_reset_enables_clean_second_pass(self):
        """After reset, a second pass must produce identical results to the first."""
        sa = StreamingAdd()

        lhs = mlx.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        rhs = mlx.array([[[10.0, 20.0, 30.0]]])

        out1 = sa.step(lhs, rhs)
        # _lhs now holds [4.0, 5.0]
        assert sa._lhs is not None

        sa.reset_state()

        out2 = sa.step(lhs, rhs)
        # Must be identical — no stale state.
        assert mlx.array_equal(out1, out2), "Second pass after reset must match first pass"


# ---------------------------------------------------------------------------
# SeanetResnetBlock.reset_state() integration test
# ---------------------------------------------------------------------------


class TestSeanetResnetBlockReset:
    """SeanetResnetBlock.reset_state() must also reset streaming_add."""

    @staticmethod
    def _make_block() -> SeanetResnetBlock:
        cfg = SeanetConfig(
            dimension=16,
            channels=1,
            causal=True,
            nfilters=16,
            nresidual_layers=1,
            ratios=[2],
            ksize=3,
            residual_ksize=3,
            last_ksize=3,
            dilation_base=2,
            pad_mode="constant",
            true_skip=True,
            compress=2,
        )
        return SeanetResnetBlock(
            cfg,
            dim=16,
            ksizes_and_dilations=[(3, 1), (1, 1)],
        )

    def test_reset_state_clears_streaming_add(self):
        block = self._make_block()
        # Manually poison the streaming_add buffers.
        block.streaming_add._lhs = mlx.ones((1, 1, 2))
        block.streaming_add._rhs = mlx.ones((1, 1, 3))

        block.reset_state()

        assert block.streaming_add._lhs is None, (
            "streaming_add._lhs must be None after SeanetResnetBlock.reset_state()"
        )
        assert block.streaming_add._rhs is None, (
            "streaming_add._rhs must be None after SeanetResnetBlock.reset_state()"
        )

    def test_step_then_reset_clears_buffers(self):
        """Run a streaming step that may leave residual, then verify reset clears it."""
        block = self._make_block()
        # Feed a single-frame input through streaming step.
        xs = mlx.zeros((1, 16, 1))
        block.step(xs)

        # Whether streaming_add has leftover or not depends on convolution padding,
        # but reset_state() must guarantee both are None regardless.
        block.reset_state()
        assert block.streaming_add._lhs is None
        assert block.streaming_add._rhs is None

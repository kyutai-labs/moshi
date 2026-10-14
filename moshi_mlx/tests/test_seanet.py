# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from itertools import product

import mlx.core as mx
import numpy as np
from moshi_mlx.models.mimi import Mimi, mimi_202407
from moshi_mlx.modules.seanet import SeanetResnetBlock, StreamingAdd


def setUpModule():
    unittest.addModuleCleanup(mx.set_default_device, mx.default_device())
    mx.set_default_device(mx.cpu)


def buffer_unmatched_samples(adder, batch_size, channels, lhs_longer):
    long = mx.full((batch_size, channels, 3), 1000.0)
    short = mx.full((batch_size, channels, 1), 100.0)
    lhs, rhs = (long, short) if lhs_longer else (short, long)
    adder.step(lhs, rhs)


def stream(step, xs):
    return mx.concat(
        [step(xs[..., index : index + 1]) for index in range(xs.shape[-1])],
        axis=-1,
    )


class StreamingResetTest(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_streaming_add_preserves_unconsumed_samples(self):
        for lhs_longer in (True, False):
            with self.subTest(lhs_longer=lhs_longer):
                adder = StreamingAdd()
                long = mx.array([[[10.0, 20.0, 30.0]]])
                short = mx.array([[[1.0]]])
                lhs, rhs = (long, short) if lhs_longer else (short, long)
                first = adder.step(lhs, rhs)
                empty = mx.zeros((1, 1, 0))
                tail = mx.array([[[2.0, 3.0]]])
                lhs, rhs = (empty, tail) if lhs_longer else (tail, empty)
                second = adder.step(lhs, rhs)
                np.testing.assert_array_equal(
                    np.array(mx.concat([first, second], axis=-1)),
                    [[[11.0, 22.0, 33.0]]],
                )

    def test_streaming_add_reset_starts_a_new_stream(self):
        for lhs_longer in (True, False):
            with self.subTest(lhs_longer=lhs_longer):
                adder = StreamingAdd()
                buffer_unmatched_samples(adder, 2, 3, lhs_longer)
                adder.reset_state()
                adder.reset_state()
                lhs = mx.random.normal((2, 3, 4))
                rhs = mx.random.normal((2, 3, 4))
                np.testing.assert_array_equal(
                    np.array(adder.step(lhs, rhs)), np.array(lhs + rhs)
                )

    def test_resnet_reset_reproduces_the_first_stream(self):
        for true_skip, lhs_longer in product((True, False), repeat=2):
            with self.subTest(true_skip=true_skip, lhs_longer=lhs_longer):
                cfg = replace(mimi_202407(2).seanet, true_skip=true_skip)
                block = SeanetResnetBlock(
                    cfg, dim=4, ksizes_and_dilations=[(3, 1), (1, 1)]
                )
                xs = mx.random.normal((2, 4, 3))
                expected = np.array(stream(block.step, xs))
                for _ in range(2):
                    buffer_unmatched_samples(block.streaming_add, 2, 4, lhs_longer)
                    block.reset_state()
                    block.reset_state()
                    np.testing.assert_allclose(
                        np.array(stream(block.step, xs)),
                        expected,
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_mimi_reset_all_reproduces_the_first_decode(self):
        cfg = mimi_202407(2)
        cfg = replace(
            cfg,
            sample_rate=16,
            frame_rate=2,
            quantizer_dim=8,
            quantizer_bins=16,
            seanet=replace(
                cfg.seanet,
                dimension=8,
                nfilters=2,
                ratios=[2, 2],
                ksize=3,
            ),
            transformer=replace(
                cfg.transformer,
                d_model=8,
                num_heads=2,
                num_layers=1,
                dim_feedforward=16,
                context=16,
                max_seq_len=32,
            ),
        )
        codes = mx.array([[[1, 2, 3], [4, 5, 6]]], dtype=mx.int32)
        for lhs_longer in (True, False):
            with self.subTest(lhs_longer=lhs_longer):
                mimi = Mimi(cfg)
                # Give the untrained codebooks nonzero embeddings without a checkpoint.
                for rvq in (mimi.quantizer.rvq_first, mimi.quantizer.rvq_rest):
                    for layer in rvq.vq.layers:
                        codebook = layer.codebook
                        codebook.embedding_sum = mx.random.normal(
                            (cfg.quantizer_bins, cfg.quantizer_dim)
                        )
                        codebook.cluster_usage = mx.ones(cfg.quantizer_bins)
                        codebook.update_in_place()
                expected = np.array(stream(mimi.decode_step, codes))
                self.assertEqual(expected.shape, (1, 1, 24))
                # Exercise reset with samples waiting on either residual branch.
                for index, layer in enumerate(mimi.decoder.layers):
                    channels = cfg.seanet.nfilters * (
                        2 ** (len(cfg.seanet.ratios) - index - 1)
                    )
                    for residual in layer.residuals:
                        buffer_unmatched_samples(
                            residual.streaming_add, 1, channels, lhs_longer
                        )
                mimi.reset_all()
                np.testing.assert_allclose(
                    np.array(stream(mimi.decode_step, codes)),
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mlx.core as mx
import mlx.nn as nn

class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._epsilon = 1e-5
        self._dim = dim

    def __call__(self, xs: mx.array) -> mx.array:
        return xs

class VectorQuantization(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, xs: mx.array) -> mx.array:
        return xs

class ResidualVectorQuantization(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, xs: mx.array) -> mx.array:
        return xs

class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, xs: mx.array) -> mx.array:
        return xs

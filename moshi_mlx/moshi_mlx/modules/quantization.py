# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .conv import Conv1d

import mlx.core as mx
import mlx.nn as nn


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._epsilon = 1e-5
        self._dim = dim
        self.initialized = mx.zeros([1], dtype=mx.float32)
        self.embedding_sum = mx.zeros([codebook_size, dim], dtype=mx.float32)
        self.cluster_usage = mx.zeros([codebook_size], dtype=mx.float32)
        cluster_usage = mx.maximum(self.cluster_usage, self._epsilon)[:, None]
        self._embedding = self.embedding_sum / cluster_usage
        self._c2 = self._embedding.square().sum(axis=-1) / 2

    def update_in_place(self):
        cluster_usage = mx.maximum(self.cluster_usage, self._epsilon)[:, None]
        self._embedding = self.embedding_sum / cluster_usage
        self._c2 = self._embedding.square().sum(axis=-1) / 2

    def update(self, parameters: dict) -> nn.Module:
        super().update(parameters)
        self.update_in_place()
        return self

    def encode(self, xs: mx.array) -> mx.array:
        target_shape = xs.shape[:-1]
        xs = xs.flatten(end_axis=-2)
        dot_prod = xs @ self._embedding.swapaxes(-1, -2)
        return (self._c2 - dot_prod).argmin(axis=-1).reshape(target_shape)

    def decode(self, xs: mx.array) -> mx.array:
        target_shape = list(xs.shape) + [self._dim]
        res = mx.take(self._embedding, xs.flatten(), axis=0).reshape(target_shape)
        return res


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim: int | None):
        super().__init__()
        codebook_dim = dim if codebook_dim is None else codebook_dim
        if dim == codebook_dim:
            self.project_in = None
            self.project_out = None
        else:
            self.project_in = nn.Linear(dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, dim)
        self.codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size)

    def encode(self, xs: mx.array) -> mx.array:
        xs = xs.swapaxes(-1, -2)
        if self.project_in is not None:
            xs = self.project_in(xs)
        return self.codebook.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        xs = self.codebook.decode(xs)
        if self.project_out is not None:
            xs = self.project_out(xs)
        return xs.swapaxes(-1, -2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, nq: int, dim: int, codebook_size: int, codebook_dim: int | None):
        super().__init__()
        layers = []
        for _ in range(nq):
            vq = VectorQuantization(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            layers.append(vq)
        self.layers = layers

    def encode(self, xs: mx.array) -> mx.array:
        codes = []
        residual = xs
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        return mx.stack(codes, axis=0)

    def decode(self, xs: mx.array) -> mx.array:
        seq_len = xs.shape[0]
        quantized = self.layers[0].decode(xs[0])
        for i in range(1, seq_len):
            quantized = quantized + self.layers[i].decode(xs[i])
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_dim: int | None,
        output_dim: int | None,
        nq: int,
        bins: int,
        force_projection: bool,
    ):
        super().__init__()
        input_dim = dim if input_dim is None else input_dim
        output_dim = dim if output_dim is None else output_dim
        if input_dim == dim and not force_projection:
            self.input_proj = None
        else:
            self.input_proj = Conv1d(input_dim, dim, 1, bias=False)
        if output_dim == dim and not force_projection:
            self.output_proj = None
        else:
            self.output_proj = Conv1d(dim, output_dim, 1, bias=False)
        self.vq = ResidualVectorQuantization(
            nq=nq,
            dim=dim,
            codebook_size=bins,
            codebook_dim=None,
        )

    def encode(self, xs: mx.array) -> mx.array:
        if self.input_proj is not None:
            xs = self.input_proj(xs)
        return self.vq.encode(xs).swapaxes(0, 1)

    def decode(self, xs: mx.array) -> mx.array:
        xs = xs.swapaxes(0, 1)
        quantized = self.vq.decode(xs)
        if self.output_proj is not None:
            quantized = self.output_proj(quantized)
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_dim: int | None,
        output_dim: int | None,
        nq: int,
        bins: int,
    ):
        super().__init__()
        self._nq = nq
        self.rvq_first = ResidualVectorQuantizer(
            dim=dim,
            input_dim=input_dim,
            output_dim=output_dim,
            nq=1,
            bins=bins,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dim=dim,
            input_dim=input_dim,
            output_dim=output_dim,
            nq=nq - 1,
            bins=bins,
            force_projection=True,
        )

    def encode(self, xs: mx.array) -> mx.array:
        codes = self.rvq_first.encode(xs)
        if self._nq > 1:
            rest_codes = self.rvq_rest.encode(xs)
            codes = mx.concat([codes, rest_codes], axis=1)
        return codes

    def decode(self, xs: mx.array) -> mx.array:
        quantized = self.rvq_first.decode(xs[:, :1])
        if self._nq > 1:
            quantized = quantized + self.rvq_rest.decode(xs[:, 1:])
        return quantized

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for all quantizers.
"""

from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):
    """Base class for quantizers."""

    def __init__(self):
        super().__init__()
        self._ema_frozen = False

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        raise NotImplementedError()

    @property
    def total_codebooks(self) -> int:
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self) -> int:
        """Number of active codebooks."""
        raise NotImplementedError()

    @property
    def semantic_quantizer(self) -> 'BaseQuantizer':
        """This returns the quantizer that models the first level of the hierarchy (typically semantic).

        In this case, it's the quantizer itself.
        """
        return self

    @property
    def acoustic_quantizer(self) -> 'BaseQuantizer':
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic).

        In this case, it's the quantizer itself.
        """
        return self

    def set_num_codebooks(self, n: int) -> None:
        """Set the number of active codebooks."""
        raise NotImplementedError()

    @property
    def ema_frozen(self) -> bool:
        """Whether to apply ema to the codebooks."""
        return self._ema_frozen

    def ema_frozen_(self, ema_frozen: bool) -> None:
        """Set whether ema should be applied to the codebooks."""
        self._ema_frozen = ema_frozen


class DummyQuantizer(BaseQuantizer):
    """Fake quantizer that actually does not perform any quantization."""

    def __init__(
        self,
        dimension: int,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
    ):
        super().__init__()
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.input_dimension == self.dimension:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )

    def forward(self, x: torch.Tensor, frame_rate: int):
        q = x.unsqueeze(1)
        x = self.output_proj(self.input_proj(x))
        return QuantizedResult(
            x, q, torch.tensor(q.numel() * 32 * frame_rate / 1000 / len(x)).to(x)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        x = self.input_proj(x)
        return x.unsqueeze(1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        y = codes.squeeze(1)
        return self.output_proj(y)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError(
            "Cannot override the number of codebooks for the dummy quantizer"
        )

    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        return 1

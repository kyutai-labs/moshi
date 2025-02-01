# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp

import torch
from torch import nn


from .base import _BaseTextConditioner, ConditionType


logger = logging.getLogger(__name__)


class TokenizedText(tp.NamedTuple):
    tokens: torch.Tensor   # should be long tensor.
    mask: torch.Tensor     # should be bool tensor.


class TextConditioner(_BaseTextConditioner[TokenizedText]):
    ...


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return TokenizedText(tokens.to(device), mask.to(device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        return ConditionType(embeds, mask)

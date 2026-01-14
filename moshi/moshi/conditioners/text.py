# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
import typing as tp

import torch
from torch import nn


from .base import _BaseTextConditioner, ConditionType


logger = logging.getLogger(__name__)


def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


class TokenizedText(tp.NamedTuple):
    tokens: torch.Tensor   # should be long tensor.
    mask: torch.Tensor     # should be bool tensor.


class TextConditioner(_BaseTextConditioner[TokenizedText]):
    ...


class Tokenizer:
    """Base tokenizer implementation
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        raise NotImplementedError()


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]

    When all possible values are known, one can use `possible_values` to provide the list
    of possible tokens. If a token doesn't exist, `pad_idx` will be used instead.
    """
    def __init__(self, n_bins: int, possible_values: list[str] | None = None):
        self.n_bins = n_bins
        self.pad_idx = n_bins
        if possible_values is None:
            self.possible_values = None
        else:
            self.possible_values = {value: idx for idx, value in enumerate(possible_values)}
            assert n_bins >= len(possible_values)

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                if self.possible_values is None:
                    output.append(hash_trick(text, self.n_bins))
                else:
                    if text not in self.possible_values:
                        raise ValueError(f"'{text}' is not in possible_values {self.possible_values}")
                    output.append(self.possible_values[text])
                lengths.append(1)

        tokens = torch.tensor(output).int()[:, None]
        mask = length_to_mask(torch.tensor(lengths))
        return TokenizedText(tokens, mask)


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, tokenizer: str, possible_values: list[str] | None = None,
                 init_scale: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.
        self.embed.weight.data *= init_scale
        if tokenizer == 'noop':
            self.tokenizer = NoopTokenizer(n_bins, possible_values)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return TokenizedText(tokens.to(device), mask.to(device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        return ConditionType(embeds, mask)

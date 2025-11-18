# Taken from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py
# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from functools import partial

import mlx.core as mx


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logits: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    """
    Apply min-p sampling to the logits.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logits: The logits from the model's output.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    # reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531-L605  # noqa

    # Softmax probabilities
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Top probability
    top_probs = mx.take_along_axis(probs, sorted_indices[..., :1], axis=-1)

    # Calculate the min_p threshold
    scaled_min_p = min_p * top_probs

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_probs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_probs = mx.where(tokens_to_remove, 0, sorted_probs)

    # Return sampled token
    sorted_token = mx.random.categorical(mx.log(selected_probs), axis=-1)
    return mx.take_along_axis(sorted_indices, sorted_token[..., None], axis=-1).squeeze(
        -1
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logprobs: mx.array,
    top_k: int,
    temperature=1.0,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.
    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    logprobs = logprobs * (1 / temperature)
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked_logprobs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        tokens selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460  # noqa
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs), axis=-1)
    token = mx.take_along_axis(sorted_indices, sorted_token[..., None], axis=-1)

    return token.squeeze(-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


@dataclass
class Sampler:
    temp: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    logit_bias: dict[int, float] | None = None

    def __call__(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        if self.logit_bias:
            indices = mx.array(list(self.logit_bias.keys()))
            values = mx.array(list(self.logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if self.temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if self.top_k is not None and self.top_k > 0:
                token = top_k_sampling(logits, self.top_k, self.temp)
            elif self.top_p > 0 and self.top_p < 1.0:
                token = top_p_sampling(logits, self.top_p, self.temp)
            elif self.min_p != 0.0:
                token = min_p_sampling(
                    logits, self.min_p, self.min_tokens_to_keep, self.temp
                )
            else:
                token = categorical_sampling(logits, self.temp)

        return token.astype(mx.int32), logprobs

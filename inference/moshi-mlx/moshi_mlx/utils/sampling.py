# Taken from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py
# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from functools import partial

import mlx.core as mx


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


@dataclass
class Sampler:
    """Top-k sampling at a temperature, as a model's `lm_gen_config` sets it; a temperature of 0
    takes the argmax."""

    temp: float
    top_k: int

    def __call__(self, logits: mx.array) -> mx.array:
        if self.temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            token = top_k_sampling(logits, self.top_k, self.temp)
        return token.astype(mx.int32)

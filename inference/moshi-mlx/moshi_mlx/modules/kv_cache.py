# Most of the code below comes from:
# https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/models/base.py#L1
# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx


class RotatingKVCache:
    """The keys and values of one layer for one-token steps: a ring of `max_size` slots, grown
    `step` slots at a time until full."""

    def __init__(self, head_dim: int, n_kv_heads: int, max_size: int, step: int = 256):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B = keys.shape[0]
        if keys.shape[2] != 1:
            raise ValueError("the cache takes one step at a time")

        # May not have hit the max size yet, so potentially keep growing the cache
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            shape = (B, self.n_kv_heads, new_size, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Rotate
        if self._idx == self.max_size:
            self._idx = 0

        # Assign
        self.keys[..., self._idx : self._idx + 1, :] = keys
        assert self.values is not None
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self):
        self.offset = 0
        self._idx = 0
        self.keys = None
        self.values = None

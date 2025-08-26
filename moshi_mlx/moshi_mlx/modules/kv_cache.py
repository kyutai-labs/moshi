# Most of the code below comes from:
# https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/models/base.py#L1
# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


class KVCache:
    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        self.offset = 0
        self.keys = None
        self.values = None

    @property
    def state(self):
        return self.keys, self.values


class RotatingKVCache:
    def __init__(self, head_dim, n_kv_heads, max_size, keep=0, step=256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Prefill mode
        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                # The largest size is self.max_size + S - 1 to ensure
                # every token gets at least self.max_size context
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        # Generation mode
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

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

    @property
    def state(self):
        return self.keys, self.values


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Any | None = None):
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask

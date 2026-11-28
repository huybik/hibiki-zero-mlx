# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .kv_cache import RotatingKVCache

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    bias_ff: bool
    bias_attn: bool
    # "rope_concat" (MLX RoPE with traditional=False) or "none"
    positional_embedding: str
    context: int
    max_period: int
    max_seq_len: int
    kv_repeat: int
    dim_feedforward: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope_concat":
            # rope_concat == interleave=False == MLX traditional=False
            self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.max_period)

    def _qkv(self, xs: mx.array, offset) -> tuple[mx.array, mx.array, mx.array]:
        # Grouped-query attention: q has num_heads, k/v have num_heads // kv_repeat.
        cfg = self.cfg
        b, t, _ = xs.shape
        H, D = cfg.num_heads, cfg.head_dim
        Hkv = H // cfg.kv_repeat
        qkv = self.in_proj(xs)
        q = qkv[..., : H * D].reshape(b, t, H, D).transpose(0, 2, 1, 3)
        k = qkv[..., H * D : H * D + Hkv * D].reshape(b, t, Hkv, D).transpose(0, 2, 1, 3)
        v = qkv[..., H * D + Hkv * D :].reshape(b, t, Hkv, D).transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        return q, k, v

    def _trim_kv(self, k: mx.array, v: mx.array, t: int) -> tuple[mx.array, mx.array]:
        k_len = k.shape[2]
        k_target_len = t + min(self.cfg.context, k_len - t)
        if k_target_len < k_len:
            k = k[:, :, k_len - k_target_len :]
            v = v[:, :, k_len - k_target_len :]
        return k, v

    def _out(self, attn: mx.array) -> mx.array:
        # Reshape the attention output back to (b, t, d_model) and project.
        b, t = attn.shape[0], attn.shape[2]
        xs = attn.transpose(0, 2, 1, 3).reshape(
            b, t, self.cfg.num_heads * self.cfg.head_dim
        )
        return self.out_proj(xs)


class MlpGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        if cfg.dim_feedforward == 4 * cfg.d_model:
            hidden = 11 * cfg.d_model // 4

        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias_ff)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.linear_in(xs)
        b, t, _ = xs.shape
        xs = xs.reshape(b, t, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.gating = MlpGating(cfg)
        self.norm1 = nn.RMSNorm(cfg.d_model, 1e-8)
        self.norm2 = nn.RMSNorm(cfg.d_model, 1e-8)
        self.self_attn = Attention(cfg)
        self.cfg = cfg
        self._pre_c = None
        self._post_c = None

    def _attn_pre(self, xs: mx.array, offset):
        # norm1 + in_proj + reshape + RoPE -> q, k, v (cache lives outside).
        return self.self_attn._qkv(self.norm1(xs), offset)

    def _attn_post(self, xs: mx.array, attn: mx.array):
        # out_proj + residual + gated FFN. The sdpa kernel (the only op touching the
        # growing KV length) runs eagerly between _pre and _post, so both compiled
        # regions see fixed shapes and compile exactly once.
        xs = xs + self.self_attn._out(attn)
        xs = xs + self.gating(self.norm2(xs))
        return xs

    def __call__(self, xs: mx.array, cache: RotatingKVCache) -> mx.array:
        # One token a step (the cache refuses more): the only data-dependent / cache-mutating
        # step (update_and_fetch) stays eager between two compiled regions of fixed shapes. The
        # RoPE offset is passed as a runtime array so growing the cache never recompiles.
        if self._pre_c is None:
            self._pre_c = mx.compile(self._attn_pre)
            self._post_c = mx.compile(self._attn_post)
        q, k, v = self._pre_c(xs, mx.array(cache.offset))
        k, v = cache.update_and_fetch(k, v)
        k, v = self.self_attn._trim_kv(k, v, 1)
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.self_attn.scale, mask=None
        )
        return self._post_c(xs, attn)


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg=cfg) for _ in range(cfg.num_layers)]

    def __call__(self, xs: mx.array, cache: list[RotatingKVCache]) -> mx.array:
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c)
        return xs

    def make_rot_cache(self) -> list[RotatingKVCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        # Cap the ring buffer at the attention context: _trim_kv discards anything
        # older anyway, so a larger buffer is dead KV memory during long mic sessions.
        return [
            RotatingKVCache(
                head_dim=self.cfg.head_dim,
                n_kv_heads=num_kv_heads,
                max_size=min(self.cfg.max_seq_len, self.cfg.context),
            )
            for _ in self.layers
        ]

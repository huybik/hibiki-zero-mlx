"""Runtime patches that make moshi_mlx (0.3.0) run hibiki-zero.

moshi_mlx targets moshi / older hibiki and misses three hibiki-zero deltas:
  1. config: `hidden_scale` is ignored (feedforward hardcoded to 4*dim) and the
     depformer feedforward is left None; `kv_repeat` is hardcoded to 1.
  2. attention: the forward pass asserts kv_repeat==1, so grouped-query
     attention (hibiki-zero main transformer uses kv_repeat=2) won't run.
  3. positional embedding: only "rope" (interleaved) is wired up; hibiki-zero
     uses "rope_concat" == RoPE with interleave=False (MLX traditional=False).

Import this module before building/loading the model.
"""
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models
from moshi_mlx.modules import transformer as T

# --- 1. config: honour hidden_scale + kv_repeat -----------------------------
_orig_from = models.LmConfig.from_config_dict.__func__


def _from_config_dict(cls, data):
    cfg = _orig_from(cls, data)
    hs = data["hidden_scale"]
    cfg.transformer.dim_feedforward = hs * data["dim"]
    cfg.depformer.transformer.dim_feedforward = hs * data["depformer_dim"]
    cfg.transformer.kv_repeat = data["kv_repeat"]
    return cfg


models.LmConfig.from_config_dict = classmethod(_from_config_dict)

# --- 2 + 3. attention: GQA + rope_concat ------------------------------------
_orig_attn_init = T.Attention.__init__


def _attn_init(self, cfg):
    _orig_attn_init(self, cfg)
    if cfg.positional_embedding in ("rope", "rope_concat"):
        # rope_concat == interleave=False == MLX traditional=False
        self.rope = nn.RoPE(
            cfg.head_dim,
            traditional=cfg.positional_embedding != "rope_concat",
            base=cfg.max_period,
        )


def _attn_call(self, xs, cache, mask=None):
    cfg = self.cfg
    b, t, _ = xs.shape
    H, D = cfg.num_heads, cfg.head_dim
    Hkv = H // cfg.kv_repeat
    qkv = self.in_proj(xs)
    q = qkv[..., : H * D].reshape(b, t, H, D).transpose(0, 2, 1, 3)
    k = qkv[..., H * D : H * D + Hkv * D].reshape(b, t, Hkv, D).transpose(0, 2, 1, 3)
    v = qkv[..., H * D + Hkv * D :].reshape(b, t, Hkv, D).transpose(0, 2, 1, 3)
    if self.rope is not None:
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
    k, v = cache.update_and_fetch(k, v)
    k_len = k.shape[2]
    k_target_len = t + min(cfg.context, k_len - t)
    if k_target_len < k_len:
        k = k[:, :, k_len - k_target_len :]
        v = v[:, :, k_len - k_target_len :]
    # mx scaled_dot_product_attention handles GQA (H a multiple of Hkv) natively.
    xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
    xs = xs.transpose(0, 2, 1, 3).reshape(b, t, H * D)
    return self.out_proj(xs)


T.Attention.__init__ = _attn_init
T.Attention.__call__ = _attn_call

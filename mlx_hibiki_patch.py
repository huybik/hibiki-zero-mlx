"""Runtime patches that make moshi_mlx (0.3.0) run hibiki-zero.

moshi_mlx targets moshi / older hibiki and misses three hibiki-zero deltas:
  1. config: `hidden_scale` is ignored (feedforward hardcoded to 4*dim) and the
     depformer feedforward is left None; `kv_repeat` is hardcoded to 1.
  2. attention: the forward pass asserts kv_repeat==1, so grouped-query
     attention (hibiki-zero main transformer uses kv_repeat=2) won't run.
  3. positional embedding: only "rope" (interleaved) is wired up; hibiki-zero
     uses "rope_concat" == RoPE with interleave=False (MLX traditional=False).
  4. depformer: hibiki-zero applies a learned per-slice output LayerNorm
     (`depformer_norms.{i}`) before each audio `linear_out`; moshi_mlx omits it,
     so the audio logits come out ~3x too small -> out-of-distribution tokens ->
     babbling/overlapping speech (the text stream is unaffected).

Import this module before building/loading the model.
"""
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models
from moshi_mlx.models import lm as L
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

# --- 4. depformer per-codebook output LayerNorm -----------------------------
# hibiki-zero applies a learned per-slice LayerNorm (`depformer_norms.{i}`,
# dim=depformer_dim, eps 1e-5, with bias) to the depformer transformer output
# *before* `linear_out` (PyTorch: logits = linears[i](depformer_norms[i](out))).
# moshi_mlx feeds the un-normalised features straight into linear_out, so the
# audio logits come out ~3x too small and uncorrelated -> babble + clipping.
# Add the norm to each slice, apply it in DepFormer.sample, and load its weights.
_orig_slice_init = L.DepFormerSlice.__init__


def _slice_init(self, in_vocab_size, out_vocab_size, main_transformer_dim,
                demux_second_stream, cfg):
    _orig_slice_init(self, in_vocab_size, out_vocab_size, main_transformer_dim,
                     demux_second_stream, cfg)
    self.norm = nn.LayerNorm(cfg.transformer.d_model, 1e-5)


L.DepFormerSlice.__init__ = _slice_init


def _depformer_sample(self, main_transformer_out, sampler, text_token, cache,
                      cfg_coef=1.0):
    tokens = []
    last_token = text_token
    for c in cache:
        c.reset()
    for slice in self.slices:
        if cfg_coef != 1:
            last_token = mx.tile(last_token, (2, 1))
        xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
        xs = slice.transformer(xs, cache=cache)
        logits = slice.linear_out(slice.norm(xs))
        if cfg_coef != 1:
            l1, l2 = logits.split(2, axis=0)
            logits = cfg_coef * l1 - (cfg_coef - 1) * l2
        last_token, _ = sampler(logits)
        tokens.append(last_token)
    return mx.stack(tokens, axis=1)


L.DepFormer.sample = _depformer_sample

# load depformer_norms.{i}.{weight,bias} into slices.{i}.norm
_orig_load = L.Lm.load_pytorch_weights


def _load_pytorch_weights(self, file, lm_config, strict=True):
    # Run the original mapping non-strict to build the rest, capture its weight
    # dict, append our depformer norms, then do the single strict load.
    pth = mx.load(file)
    extra = {}
    for i in range(lm_config.depformer.num_slices):
        for p in ("weight", "bias"):
            k = f"depformer_norms.{i}.{p}"
            if k in pth:
                extra[f"depformer.slices.{i}.norm.{p}"] = pth[k]
    captured = {}
    real_load = self.load_weights

    def _capture(items, strict):
        captured.update(dict(items))
        return None

    self.load_weights = _capture
    try:
        _orig_load(self, file, lm_config, strict=False)
    finally:
        self.load_weights = real_load
    captured.update(extra)
    return self.load_weights(list(captured.items()), strict=strict)


L.Lm.load_pytorch_weights = _load_pytorch_weights

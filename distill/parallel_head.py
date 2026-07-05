"""Parallel codebook head (Track B, distill_plan §3).

Replaces the AR depformer's N sequential slices with ONE forward that emits all
N codebooks at once. Delay-pattern design: codebook k conditions on the already
-decided token of codebook k from the *previous* frame (cross-frame, so within a
frame everything is parallel — no left-to-right recurrence). A small shared
trunk (bidirectional over the N codebook positions) lets the codebooks share
information in that single pass. `num_passes` > 1 does MaskGIT-style iterative
refinement (re-condition on the current frame's provisional argmax) — placeholder
for later tuning; default 1.

The per-codebook projections/embeddings/norms are stored as *stacked* arrays and
applied with batched einsum / grouped ops, so the whole head is a handful of GPU
launches (vs the AR head's ~370). Warm start (distill_plan §5): the vocab-facing
w_in / w_out / norm and the token/text embeddings are copied from the AR
depformer where shapes line up; the trunk is trained from scratch.
"""
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ParallelHeadConfig:
    num_codebooks: int      # dep_q: 16 (3B) / 8 (1B)
    main_dim: int           # transformer_out dim: 2048
    head_dim: int           # depformer dim: 1024
    audio_out_vocab: int    # audio_vocab_size - 1: 2048
    audio_in_vocab: int     # audio_vocab_size: 2049 (includes pad token)
    text_in_vocab: int      # 48001
    audio_pad_token: int    # audio_vocab_size - 1: 2048
    num_layers: int = 6
    num_heads: int = 16
    num_passes: int = 1


class _TrunkLayer(nn.Module):
    """Norm-first gated transformer layer, bidirectional (no cache, no RoPE)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.RMSNorm(dim, 1e-8)
        self.norm2 = nn.RMSNorm(dim, 1e-8)
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        hidden = 11 * dim // 4
        self.gate_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.gate_out = nn.Linear(hidden, dim, bias=False)

    def _attn(self, xs: mx.array) -> mx.array:
        b, t, d = xs.shape
        H, D = self.num_heads, self.head_dim
        q, k, v = self.in_proj(xs).reshape(b, t, 3, H, D).transpose(2, 0, 3, 1, 4)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self.out_proj(a.transpose(0, 2, 1, 3).reshape(b, t, d))

    def _mlp(self, xs: mx.array) -> mx.array:
        h = self.gate_in(xs)
        b, t, _ = h.shape
        h = h.reshape(b, t, 2, -1)
        return self.gate_out(nn.silu(h[:, :, 0]) * h[:, :, 1])

    def __call__(self, xs: mx.array) -> mx.array:
        xs = xs + self._attn(self.norm1(xs))
        xs = xs + self._mlp(self.norm2(xs))
        return xs


class ParallelHead(nn.Module):
    def __init__(self, cfg: ParallelHeadConfig):
        super().__init__()
        self.cfg = cfg
        N, hd = cfg.num_codebooks, cfg.head_dim
        bi = cfg.main_dim ** -0.5
        bo = hd ** -0.5
        self.w_in = mx.random.uniform(-bi, bi, (N, hd, cfg.main_dim))      # bnh<-bm
        self.w_out = mx.random.uniform(-bo, bo, (N, cfg.audio_out_vocab, hd))  # bnv<-bnh
        self.cb_emb = mx.random.normal((N, cfg.audio_in_vocab, hd)) * bo
        self.text_emb = nn.Embedding(cfg.text_in_vocab, hd)
        self.pos_emb = mx.zeros((N, hd))
        self.norm_w = mx.ones((N, hd))
        self.norm_b = mx.zeros((N, hd))
        self.trunk = [_TrunkLayer(hd, cfg.num_heads) for _ in range(cfg.num_layers)]
        self._prev = None       # streaming state (inference only; not a parameter)
        self._fwd_c = None

    def _group_ln(self, xs: mx.array) -> mx.array:
        mu = xs.mean(-1, keepdims=True)
        var = xs.var(-1, keepdims=True)
        xn = (xs - mu) * mx.rsqrt(var + 1e-5)
        return xn * self.norm_w + self.norm_b

    def _forward_logits(self, transformer_out, text_token, cond_tokens):
        # transformer_out (B, main_dim); text_token (B,); cond_tokens (B, N)
        N = self.cfg.num_codebooks
        base = mx.einsum("bm,nhm->bnh", transformer_out, self.w_in)
        cond = self.cb_emb[mx.arange(N)[None], cond_tokens]      # (B, N, hd)
        te = self.text_emb(text_token)[:, None, :]               # (B, 1, hd)
        xs = base + cond + self.pos_emb + te
        for layer in self.trunk:
            xs = layer(xs)
        xs = self._group_ln(xs)
        return mx.einsum("bnh,nvh->bnv", xs, self.w_out)         # (B, N, out_vocab)

    def __call__(self, transformer_out, text_token, prev_tokens):
        """Training entry point. Returns logits (B, N, out_vocab)."""
        cond = prev_tokens
        logits = None
        for p in range(self.cfg.num_passes):
            logits = self._forward_logits(transformer_out, text_token, cond)
            if p < self.cfg.num_passes - 1:
                cond = mx.argmax(logits, axis=-1)                # provisional refine
        return logits

    def sample(self, transformer_out, text_token, sampler):
        """Streaming inference. transformer_out (B,1,main_dim), text_token (B,1);
        manages its own prev-frame token state; returns (B, N, 1)."""
        B = transformer_out.shape[0]
        N = self.cfg.num_codebooks
        if self._prev is None:
            self._prev = mx.full((B, N), self.cfg.audio_pad_token, dtype=mx.int32)
        to = transformer_out.reshape(B, -1)
        tt = text_token.reshape(B)
        if self.cfg.num_passes == 1:
            if self._fwd_c is None:
                self._fwd_c = mx.compile(self._forward_logits)  # fixed shapes -> 1 compile
            logits = self._fwd_c(to, tt, self._prev)
        else:
            logits = self(to, tt, self._prev)
        tok, _ = sampler(logits)                                 # (B, N)
        self._prev = tok.astype(mx.int32)
        return tok[:, :, None]

    def reset(self):
        self._prev = None


def build_head(lm_config, num_passes: int = 1) -> ParallelHead:
    cfg = ParallelHeadConfig(
        num_codebooks=lm_config.depformer.num_slices,
        main_dim=lm_config.transformer.d_model,
        head_dim=lm_config.depformer.transformer.d_model,
        audio_out_vocab=lm_config.audio_vocab_size - 1,
        audio_in_vocab=lm_config.audio_vocab_size,
        text_in_vocab=lm_config.text_in_vocab_size,
        audio_pad_token=lm_config.audio_padding_token,
        num_layers=lm_config.depformer.transformer.num_layers,
        num_heads=lm_config.depformer.transformer.num_heads,
        num_passes=num_passes,
    )
    return ParallelHead(cfg)


def warm_start(head: ParallelHead, bf16_path: str) -> None:
    """Copy AR depformer weights (MLX-named bf16 checkpoint) into the head where
    shapes line up (distill_plan §5): stacked w_in / w_out / norm and the
    token/text embeddings. The trunk is left freshly initialised."""
    w = mx.load(str(bf16_path))
    N = head.cfg.num_codebooks

    def stack(keyf, ref):
        rows = []
        for k in range(N):
            v = w.get(keyf(k))
            rows.append(mx.array(v.astype(mx.float32))
                        if v is not None and tuple(v.shape) == tuple(ref[k].shape)
                        else ref[k])
        return mx.stack(rows)

    head.w_in = stack(lambda k: f"depformer.slices.{k}.linear_in.weight", head.w_in)
    head.w_out = stack(lambda k: f"depformer.slices.{k}.linear_out.weight", head.w_out)
    head.norm_w = stack(lambda k: f"depformer.slices.{k}.norm.weight", head.norm_w)
    head.norm_b = stack(lambda k: f"depformer.slices.{k}.norm.bias", head.norm_b)
    # per-codebook token emb: slice k+1 embeds audio codebook k (last one has none).
    head.cb_emb = stack(lambda k: f"depformer.slices.{k + 1}.emb.weight", head.cb_emb)
    te = w.get("depformer.slices.0.emb.weight")
    if te is not None and tuple(te.shape) == tuple(head.text_emb.weight.shape):
        head.text_emb.weight = mx.array(te.astype(mx.float32))

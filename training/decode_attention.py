"""One query a slot over the first `lengths[b]` keys of its own preallocated cache, in Triton: frame.py's
backbone attention, and the SFT trainer's generation step (lm_training). Each program reads only its slot's live
keys and values (the dense masked form reads the whole cache every frame), and the length is read on the device, so
the kernel replays inside a CUDA graph. The query heads sharing a key head go through the tensor cores together,
padded to 16 rows."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

ROWS = 16  # tl.dot's smallest M


@triton.jit
def _attend(Q, K, V, O, LEN, scale, s_qb, s_qh, s_kb, s_kh, s_kl, s_ob, s_oh,
            G: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    n = tl.load(LEN + b)
    g, d = tl.arange(0, ROWS), tl.arange(0, D)
    live = g < G
    q = tl.load(Q + b * s_qb + (h * G + g)[:, None] * s_qh + d[None, :], mask=live[:, None], other=0.0)
    m = tl.full([ROWS], float("-inf"), tl.float32)
    total = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    base = b * s_kb + h * s_kh
    for start in range(0, n, BLOCK):
        pos = start + tl.arange(0, BLOCK)
        valid = pos < n
        at = base + pos[:, None] * s_kl + d[None, :]
        k = tl.load(K + at, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - new[:, None])
        keep = tl.exp(m - new)
        v = tl.load(V + at, mask=valid[:, None], other=0.0)
        total = total * keep + tl.sum(p, axis=1)
        acc = acc * keep[:, None] + tl.dot(p.to(v.dtype), v)
        m = new
    out = acc / total[:, None]
    tl.store(O + b * s_ob + (h * G + g)[:, None] * s_oh + d[None, :], out.to(O.dtype.element_ty), mask=live[:, None])


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor, scale: float, *,
           block: int = 64, warps: int = 4, stages: int = 2) -> torch.Tensor:
    """q [B, Hq, D], k and v [B, Hkv, L, D] (each slot's cache, contiguous in its last axis), lengths [B] (keys
    0..lengths-1 attended; at least 1) -> [B, Hq, D]. Hq a multiple of Hkv, at most 16 a key head."""
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv or Hq // Hkv > ROWS:
        raise ValueError(f"{Hq} query heads over {Hkv} key heads: attend takes a multiple, at most {ROWS} a key head")
    if k.shape != v.shape or k.stride() != v.stride() or k.stride(3) != 1 or q.stride(2) != 1:
        raise ValueError("attend: k and v share one layout, contiguous in the last axis, as q")
    out = torch.empty_like(q)
    _attend[(B, Hkv)](q, k, v, out, lengths, scale, q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                      out.stride(0), out.stride(1), G=Hq // Hkv, D=D, BLOCK=block, ROWS=ROWS, num_warps=warps,
                      num_stages=stages)
    return out

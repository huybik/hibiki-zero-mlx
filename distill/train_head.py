#!/usr/bin/env python
"""Distill the parallel head from the teacher dump (Track B / distill_plan §5, P3).

Loss = KL(teacher || student, temperature T) + CE on the teacher's sampled tokens,
optimising the HEAD PARAMETERS ONLY (the head is a standalone module; the frozen
main is never in the graph). Adam, clip-level train/val split, head-only
checkpoints.

  # full smoke train (loss should fall):
  python distill/train_head.py --teacher distill/teacher_3b --steps 400 \
      --warm-start weights/hibiki.bf16.safetensors --out distill/head_3b.safetensors
  # capacity/gradient sanity (overfit one tiny shard to ~0):
  python distill/train_head.py --teacher distill/teacher_3b --overfit --steps 300
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from moshi_mlx import models
from distill.parallel_head import build_head, warm_start

ROOT = Path(__file__).resolve().parent.parent


def load_shards(teacher_dir: Path, pad_token: int, max_shards: int | None = None):
    """Return per-clip arrays: transformer_out, text_tokens, prev_tokens (delay-1
    conditioning), teacher_tokens, teacher_logits. Concatenated across clips with
    a `clip_ids` array so train/val split is clip-clean."""
    state = json.loads((teacher_dir / "manifest.json").read_text())
    shards = state["shards"][: max_shards] if max_shards else state["shards"]
    tout, ttok, prev, ctok, clog, clip_ids = [], [], [], [], [], []
    cid = 0
    for sh in shards:
        d = mx.load(str(teacher_dir / sh["file"]))
        lens = np.asarray(d["clip_lengths"]).tolist()
        to = np.asarray(d["transformer_out"]); tt = np.asarray(d["text_tokens"])
        cc = np.asarray(d["teacher_tokens"]); cl = np.asarray(d["teacher_logits"])
        off = 0
        for L in lens:
            sl = slice(off, off + L)
            ct = cc[sl]
            pv = np.empty_like(ct)
            pv[0] = pad_token
            pv[1:] = ct[:-1]                     # delay-1: prev frame's tokens
            tout.append(to[sl]); ttok.append(tt[sl]); prev.append(pv)
            ctok.append(ct); clog.append(cl[sl])
            clip_ids.append(np.full(L, cid, np.int32)); cid += 1
            off += L
    return {
        "transformer_out": np.concatenate(tout),
        "text_tokens": np.concatenate(ttok),
        "prev_tokens": np.concatenate(prev),
        "teacher_tokens": np.concatenate(ctok),
        "teacher_logits": np.concatenate(clog),
        "clip_ids": np.concatenate(clip_ids),
        "num_clips": cid,
    }


def loss_fn(head, to, tt, prev, tgt_tok, tgt_logits, temp, kl_w, ce_w):
    logits = head(to, tt, prev)                              # (B, N, V) f32
    logits = logits.astype(mx.float32)
    tgt_logits = tgt_logits.astype(mx.float32)
    # KL(teacher || student) * T^2 (Hinton distillation).
    t_logp = nn.log_softmax(tgt_logits / temp, axis=-1)
    s_logp = nn.log_softmax(logits / temp, axis=-1)
    kl = (mx.exp(t_logp) * (t_logp - s_logp)).sum(-1).mean() * (temp * temp)
    # CE on teacher's sampled tokens.
    ce = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt_tok.reshape(-1), reduction="mean"
    )
    return kl_w * kl + ce_w * ce, (kl, ce)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="distill/teacher_3b")
    p.add_argument("--model", default="3b", help="config source for head dims")
    p.add_argument("--out", default="distill/head_3b.safetensors")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--warm-start", default=None, help="bf16 checkpoint for warm init")
    p.add_argument("--overfit", action="store_true",
                   help="train+eval on the first --overfit-frames frames to ~0 loss")
    p.add_argument("--overfit-frames", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    mx.random.seed(299792458)
    cfg = models.LmConfig.from_config_dict(
        json.loads((pl_dir(args.model) / "config.json").read_text()))
    head = build_head(cfg, num_passes=args.passes)
    if args.warm_start:
        warm_start(head, args.warm_start)
    mx.eval(head.parameters())
    nparams = sum(v.size for _, v in tree_flatten(head.parameters()))
    print(f"head: {cfg.depformer.num_slices} cb, {args.passes} pass(es), "
          f"{nparams/1e6:.1f}M params")

    data = load_shards(ROOT / args.teacher, cfg.audio_padding_token)
    n = len(data["text_tokens"])
    if args.overfit:
        tr = va = np.arange(min(args.overfit_frames, n))
        print(f"overfit mode: {len(tr)} frames")
    else:
        rng = np.random.default_rng(0)
        val_clips = set(rng.choice(data["num_clips"],
                        max(1, int(args.val_frac * data["num_clips"])), replace=False))
        is_val = np.array([c in val_clips for c in data["clip_ids"]])
        tr = np.where(~is_val)[0]; va = np.where(is_val)[0]
        print(f"data: {n} frames ({data['num_clips']} clips) -> "
              f"train {len(tr)} / val {len(va)}")

    opt = optim.Adam(learning_rate=args.lr)
    lag = nn.value_and_grad(head, loss_fn)

    def batch(idx):
        return (mx.array(data["transformer_out"][idx].astype(np.float32)),
                mx.array(data["text_tokens"][idx].astype(np.int32)),
                mx.array(data["prev_tokens"][idx].astype(np.int32)),
                mx.array(data["teacher_tokens"][idx].astype(np.int32)),
                mx.array(data["teacher_logits"][idx].astype(np.float32)))

    def evaluate():
        tot = 0.0; nb = 0
        for s in range(0, len(va), args.batch):
            idx = va[s:s + args.batch]
            to, tt, pv, tk, tl = batch(idx)
            l, _ = loss_fn(head, to, tt, pv, tk, tl,
                           args.temp, args.kl_weight, args.ce_weight)
            mx.eval(l)
            tot += float(l); nb += 1
        return tot / max(nb, 1)

    rng = np.random.default_rng(1)
    curve = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = rng.choice(tr, min(args.batch, len(tr)), replace=len(tr) < args.batch)
        to, tt, pv, tk, tl = batch(idx)
        (loss, (kl, ce)), grads = lag(head, to, tt, pv, tk, tl,
                                      args.temp, args.kl_weight, args.ce_weight)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state, loss)
        if step % args.log_every == 0 or step == 1:
            curve.append((step, float(loss), float(kl), float(ce)))
            print(f"step {step:4d}  loss {float(loss):.4f}  kl {float(kl):.4f}  "
                  f"ce {float(ce):.4f}")
    val = evaluate()
    dt = time.perf_counter() - t0
    print(f"final val loss {val:.4f}  ({dt:.1f}s, {args.steps} steps)")

    flat = dict(tree_flatten(head.parameters()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.out, {k: v.astype(mx.float32) for k, v in flat.items()},
                        metadata={"passes": str(args.passes), "model": args.model})
    print(f"saved head -> {args.out}")
    print("curve:", " ".join(f"{s}:{l:.3f}" for s, l, _, _ in curve))


def pl_dir(model):
    from hibiki_mlx import pipeline as pl
    return pl.resolve_weights_dir(model)


if __name__ == "__main__":
    main()

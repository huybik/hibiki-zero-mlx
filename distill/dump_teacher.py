#!/usr/bin/env python
"""Teacher dump (Track B / distill_plan §5, P1) — INFERENCE ONLY.

Runs the FROZEN model over source audio (CoVoST2 fr wavs) and caches, per frame:
  transformer_out (main_dim, fp16), the sampled text token, the N teacher
  codebook LOGITS (fp16) and the sampled teacher tokens. Resumable shards +
  manifest. The parallel head is later distilled against this cache; the frozen
  main is never touched (distill_plan §4).

  python distill/dump_teacher.py --model 3b --num-clips 50
  python distill/dump_teacher.py --model 3b --hours 3 --out distill/teacher_3b

Disk: full-vocab logits are stored fp16 => ~ N*vocab*2 bytes/frame (3B: 16*2048*2
= 64 KB) + transformer_out (main_dim*2). Smoke (~50 clips ~ 3-4k frames) ~= 0.3 GB.
To scale past ~5 h, switch to top-k logits (see reports/parallel_head_smoke.md).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sphn

from hibiki_mlx import pipeline as pl
from moshi_mlx import models, utils

ROOT = Path(__file__).resolve().parent.parent


def _reset(model):
    for c in model.transformer_cache:
        c.reset()


def dump(model, lm_config, ct, mimi_enc, wav: str):
    """Run one clip; return dict of per-frame numpy arrays (frames in order)."""
    other_cb = lm_config.other_codebooks
    in_pcms, _ = sphn.read(wav, sample_rate=24000)
    steps = in_pcms.shape[-1] // 1920
    _reset(model)
    gen = models.LmGen(
        model=model, max_steps=steps + 8,
        text_sampler=utils.Sampler(top_k=25, temp=0.4),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0, check=False,
    )
    tout, ttok, ctok, clog = [], [], [], []
    for idx in range(steps):
        pcm = in_pcms[None, 0:1, idx * 1920:(idx + 1) * 1920]
        codes = mimi_enc.encode_step(pcm)
        oat = np.transpose(codes, (0, 2, 1))[0, :, :other_cb]
        text_tokens, transformer_out = gen.step(mx.array(oat), ct)
        logits = model.depformer.last_logits      # (1, N, vocab)
        tokens = model.depformer.last_tokens       # (1, N, 1)
        mx.eval(text_tokens, transformer_out, logits, tokens)
        tout.append(np.asarray(transformer_out.reshape(-1).astype(mx.float16)))
        ttok.append(int(text_tokens.reshape(-1)[0].item()))
        ctok.append(np.asarray(tokens.reshape(-1).astype(mx.int32)))
        clog.append(np.asarray(logits[0].astype(mx.float16)))
    if not tout:
        return None
    return {
        "transformer_out": np.stack(tout),                       # (T, main_dim) f16
        "text_tokens": np.array(ttok, dtype=np.int32),           # (T,)
        "teacher_tokens": np.stack(ctok).astype(np.int32),       # (T, N)
        "teacher_logits": np.stack(clog),                        # (T, N, vocab) f16
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="3b")
    p.add_argument("--quant", default="q4", choices=["q4", "q4-depq3"])
    p.add_argument("--manifest", default="remote_dataset/covost2_fr_en_test/manifest.csv")
    p.add_argument("--out", default=None, help="shard dir (default distill/teacher_<model>)")
    p.add_argument("--num-clips", type=int, default=50)
    p.add_argument("--hours", type=float, default=None, help="bound by source hours instead")
    p.add_argument("--shard-frames", type=int, default=4000)
    args = p.parse_args()

    out = Path(args.out or ROOT / "distill" / f"teacher_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    state = json.loads(man_path.read_text()) if man_path.exists() else {
        "model": args.model, "done": [], "shards": [], "frames": 0}
    done = set(state["done"])

    rows = list(csv.DictReader((ROOT / args.manifest).open()))
    budget_s = args.hours * 3600 if args.hours else None

    weights_dir = pl.resolve_weights_dir(args.model)
    model, lm_config, _tok, mimi_enc, _dec = pl.load(weights_dir, args.quant)
    model.depformer.capture = True
    ct = None
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    model.warmup(ct)

    buf: dict[str, list] = {}
    clip_lens: list[int] = []
    used_s = 0.0
    n_new = 0
    t0 = time.perf_counter()

    def flush():
        if not clip_lens:
            return
        idx = len(state["shards"])
        arrs = {k: mx.array(np.concatenate(v)) for k, v in buf.items()}
        arrs["clip_lengths"] = mx.array(np.array(clip_lens, dtype=np.int32))
        name = f"shard_{idx:04d}.safetensors"
        mx.save_safetensors(str(out / name), arrs)
        frames = int(sum(clip_lens))
        state["shards"].append({"file": name, "frames": frames, "clips": len(clip_lens)})
        state["frames"] += frames
        man_path.write_text(json.dumps(state, indent=2))
        print(f"  wrote {name}: {len(clip_lens)} clips, {frames} frames")
        buf.clear(); clip_lens.clear()

    for row in rows:
        wav = row["audio_file"]
        if wav in done:
            continue
        if args.hours is not None:
            if used_s >= budget_s:
                break
        elif n_new >= args.num_clips:
            break
        wav_path = wav if Path(wav).is_absolute() else str(ROOT / wav)
        d = dump(model, lm_config, ct, mimi_enc, wav_path)
        done.add(wav); state["done"].append(wav)
        if d is None:
            continue
        for k, v in d.items():
            buf.setdefault(k, []).append(v)
        clip_lens.append(len(d["text_tokens"]))
        used_s += float(row.get("duration_s", 0) or 0)
        n_new += 1
        print(f"[{n_new}] {Path(wav).name}: {len(d['text_tokens'])} frames "
              f"(total {state['frames'] + int(sum(clip_lens))} frames)")
        if sum(clip_lens) >= args.shard_frames:
            flush()
    flush()
    dt = time.perf_counter() - t0
    print(f"\ndone: {state['frames']} frames across {len(state['shards'])} shards "
          f"in {out} ({dt:.1f}s, {used_s/60:.1f} min audio)")


if __name__ == "__main__":
    main()

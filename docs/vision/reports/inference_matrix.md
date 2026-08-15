# Phase 2 — Inference optimization sweep (model x quant matrix)

All numbers measured on M4 Pro with `scripts/bench.py` (150 timed frames of `leon.wav`,
15 warmup frames, fixed seed, mx.eval barriers between stages) unless noted. iPhone
numbers are projections: `--scale` = assumed A18 Pro / M4 Pro GPU throughput ratio for
these kernels, default **0.5** (plausible band 0.4–0.6). No device was in the loop.

## The matrix

| config | main ms | depformer ms | LM total ms | LM size | live RT (M4) | silence-in | CoVoST2 fr→en n=30 |
|---|---|---|---|---|---|---|---|
| 3B q4 (pre-S2 baseline) | 11.8 | 12.4 (16×0.78) | 24.3 | 2.41 GB | 3.3× | PASS .0002/.012 | BLEU 25.7 / chrF 49.4 * |
| **3B q4 (+S2)** | 11.9 | 10.2 (16×0.64) | **22.1** | 2.41 GB | 3.6× | PASS .0002/.012 | (same weights as baseline) |
| 3B q4-depq3 (+S2) | 11.9 | 9.5 | 21.4 | 2.21 GB | 3.7× | PASS .0002/.012 | leon coherent, early-stop OK |
| 1B q4 (+S2) | 7.4 | 7.2 (8×0.90) | **14.7** | 1.13 GB | 5.4× | PASS .0002/.013 | **BLEU 28.4 / chrF 54.6** |
| 1B q4-depq3 (+S2) | 7.4 | 7.3 | 14.7 | 1.05 GB | 5.4× | PASS .045/.81 ** | BLEU 26.8 / chrF 53.7 |

\* 3B BLEU is the standing Phase-1 reference (same q4 weights; not rerun).
\** Seed-dependent: the bench seed produces some audible output on zeros (rms 0.045),
a second seed gives healthy 0.0004/0.014. Both inside the gate (rms<0.10, peak<1.1).

Mimi codec (CPU, rustymimi): encode ~17.9 ms, decode ~16.7 ms per frame for both models —
fully hidden by the 3-thread pipeline; the live critical path is the LM step alone.
File-mode pipelined throughput: 3B ~3.0× RT, 1B ~4.1× RT on leon.

## Strategy outcomes

**S1 — Hibiki-M 1B (phone candidate): KEPT, primary.** Wired into the runtime
(`hibiki_mlx.load()` honors each model's config: dep_q 8 vs 16, n_q 16 vs 32, delays,
conditioners; `main.py`/`run_batch.py --model {3b,1b}`). LM step 14.7 ms vs 22.1 ms
(3B). Quality is *better* than the 3B on FR→EN (BLEU 28.4 vs 25.7) — Hibiki-M is the
dedicated FR→EN model, while the 3B is the multilingual zero-shot one. Caveat: 1B is
FR→EN only; the 3B stays the Mac/teacher and the FR/ES/PT/DE model.

**S2 — Depformer launch reduction: KEPT (−18% depformer, −9% LM frame on 3B).**
`DepFormer.sample` (cfg_coef==1 path) now threads KV functionally through one
`mx.compile`d step per slice (fixed shapes → each slice compiles once), replacing the
shared-KVCache reset/realloc machinery (which re-allocated 256-wide zero buffers per
layer per frame). 3B: dep 12.4→10.2 ms, LM 24.3→22.1 ms. 1B: LM 15.4→14.7 ms (−5%).
Output not byte-identical to the old path (compile changes op order in bf16) but leon
is coherent and all gates pass. The 16 sequential slice passes remain — only the
Track B parallel head removes them.

**S3 — Quant matrix: depq3 KEPT with a narrower predicate than planned.**
Quantizing the *whole* depformer to q3 (incl. slice embeddings + linear_out) makes the
3B babble/loop through the tail flush on both seeds tried (text degenerates into
"The Olympic. The Olympic. …" and the pad early-stop never fires) — negative result;
the old "depformer q3 quality-safe" note did not cover embeddings/output heads.
Shipped `q4-depq3` = q3 on the slice *transformers* only: 3B 2.41→2.21 GB, 1B
1.13→1.05 GB, speed unchanged on M4 (launch-bound, as expected), silence PASS, leon
coherent on both, 1B depq3 CoVoST BLEU 26.8 (−1.6 vs q4). Pure size/bandwidth win for
the phone; on M4 there is no speed reason to prefer it.

**S4 — KV-cache discipline: capped.** `Transformer.make_rot_cache` now caps
`RotatingKVCache.max_size` at the model's attention context (attention already trimmed
to context, so anything larger was dead memory). Live-window KV at cap (bf16):

| model | context (window) | KV at cap | was (4096 cap) |
|---|---|---|---|
| 3B (28L, 8 kv-heads GQA, d128) | 3000 fr = 4.0 min | **344 MB** | 470 MB |
| 1B (16L, 16 kv-heads, d128) | 500 fr = 40 s | **66 MB** | 537 MB |

Verified past rotation (1B, 520 frames > 500 context: stable speed, no crash).
**int8-KV is deliberately not implemented**: KV read bandwidth only matters once the
launch-bound AR depformer is replaced (post-parallel-head work). It would halve the
numbers above.

**S5 — Unified harness: `scripts/bench.py`** (replaces `profile_mlx.py`, deleted).
Per-stage table, RT factors, artifact size, projected iPhone frame time (`--scale`),
and the silence-in gate (`--silence`, exit code = gate result).

## Chosen phone config

**Hibiki-M 1B q4-gs32 (`weights/hibiki-m-mlx-q4`), with q4-depq3 as the
size-constrained option.** Reasoning: best measured quality on the language pair
(BLEU 28.4), smallest LM that passes all gates (1.13 GB; 1.05 GB depq3 at −1.6 BLEU),
and the largest margin under the phone budget. The 3B remains the Mac/teacher model
(multilingual, distill source); it is not the phone artifact.

### Projected iPhone frame time vs the 80 ms budget (AR head)

LM step, `iphone_ms = m4_ms / scale`:

| config | M4 ms | scale 0.6 | scale 0.5 (default) | scale 0.4 |
|---|---|---|---|---|
| 1B q4 | 14.7 | 24.5 | **29.4** | 36.7 |
| 3B q4 | 22.1 | 36.8 | 44.2 | 55.3 |

The codec runs on CPU threads (enc 17.9 + dec 16.7 ms on M4) and is off the critical
path if the phone app keeps the 3-thread pipeline; even at 0.5× CPU scale each codec
stage (~36 ms) stays under one 80 ms frame. So the **1B AR config already projects
inside the budget at every scale assumption (~29 ms @0.5, worst case ~37 ms)**; the 3B
fits only the LM step and only with slim headroom at pessimistic scale.

### Gap for Track B (parallel codebook head)

The depformer is still ~half the LM step (1B: 7.3 of 14.7 ms; 3B: 10.2 of 22.1 ms) and
is sequential-launch-bound — S2 shaved dispatch overhead but cannot remove the 8/16
serial passes, and quant does nothing for it on-GPU. Track B replacing it with a
~1–4 ms parallel head projects:

- 1B: LM 14.7 → ~8.5–11 ms M4 → **~17–27 ms iPhone** — comfortable headroom for
  lower-end phones, thermals, and battery.
- 3B: LM 22.1 → ~13–16 ms M4 → ~26–40 ms iPhone — what would make the *3B* phone-viable.

That is the gap Track B must close: not feasibility of the 1B (already projected
in-budget), but the margin that survives a real A-series device, and the 3B option.

# SFT training speed

Read when touching `training/finetune/lm_training.py`, or when choosing a GPU pod or measuring
training, validation or checkpoint speed.
Back to [CONTEXT.md](../../CONTEXT.md) · [Fast path](#fast-path) ·
[Measurements](#measurements) · [By GPU](#by-gpu)

The trainer itself is in [sft-training](sft-training.md); the pod procedure in
[docs/finetune.md](../finetune.md).

## Fast path

`lm_training.install()` (run when `common` is imported) replaces five things in moshi 0.2.13
with the same math on the same weights:

- the causal attention mask becomes `is_causal=True` (flash);
- grouped-query attention runs in SDPA (`enable_gqa`) instead of copying each key and value to 16
  heads: in generation that copy is the whole KV cache, twice per layer and step;
- the depformer's training forward runs its 16 codebook steps step-major (`unbind`/`stack`,
  whose backward is one copy, instead of slices whose backward zero-fills a full gradient each),
  compiled with dynamic shapes, with the memory-efficient attention kernel;
- the training gating FFN splits with `unbind` for the same reason;
- the main transformer's training forward (training mode, no streaming state) runs its 28 layers
  as one function compiled with dynamic shapes, moshi's RMS norms and rope (at offset 0) inline
  and `is_causal` attention, which equals moshi's mask while T is within the 3,000-frame
  context (`train.py` checks `--stream-seconds` at startup, the forward each call). Compiling
  moshi's own forward instead recompiles per batch shape. Validation and generation (eval mode)
  keep moshi's forward;
- each compiled function keeps up to 64 graphs, not dynamo's 8. moshi compiles `_rms_norm`,
  `apply_rope` and `gating_forward_kernel` once per dtype, rank, grad mode, autocast state and
  shape class, and past the limit a function's new variants run eagerly for the rest of the
  process. Training builds two graphs of `_rms_norm` (the LM's output norm); a process that
  resumes, validates, trains and validates again built 8 (2026-11-03, H200 NVL).

Checked against moshi as shipped (2026-11-03, H100 NVL): fp32 loss equal, logits within 1.7e-6
and every gradient within 3.1e-6; in bf16 both are as far from an fp32 reference (gradients:
median 8.5e-3, max 9.4e-2); fp32 greedy decoding gives the same tokens for 494 steps on 16
FLEURS rows. Sampled decoding diverges from moshi's after a few words (any rounding change
does that), with outputs of the same kind. With the compiled main transformer (H200 NVL): fp32
loss equal (9.301917), logits within 2.6e-6 and every gradient within 1.2e-5; in bf16 its
gradients are a median 7.8e-3 (max 9.9e-2) from the fp32 reference, moshi's own 8.6e-3
(1.3e-1).

## Measurements

Measured on one H100 NVL (400 W, at its power limit) on 16 shards' real batches, 2026-11-03:

| Training step | frames/s | s/step | peak |
|---|---:|---:|---:|
| Torch 2.8, moshi as shipped | 6,930 | 0.595 | 76.8 GiB |
| Torch 2.15, moshi as shipped | 7,715 | 0.528 | 76.8 GiB |
| + step-major depformer and gating | 9,480 | 0.430 | 76.7 GiB |
| + compiled depformer, GQA, 4 data workers | 10,669 | 0.385 | 73.6 GiB |

- The 2.8 row ran on another 16-shard sample; per-frame rates compare.
- Tried and dropped: flash (8,880 frames/s) or cuDNN (8,969) for the depformer's 16-position
  attention, one `bmm` per step group instead of per-step linears (no gain).
- Tried on the 700 W H100 SXM (2026-11-03; 200 warm-up steps, then 80, against 13,405):
  - torchao 0.18 float8 on the main transformer's 112 linears: eager 8,587 (tensorwise) and
    8,050 (rowwise); compiled with the transformer 13,920, and the loss moved (4.006, not 4.018);
  - math attention in the depformer 13,381; clipping inside fused AdamW through its
    `grad_scale` (GradScaler's private hook) 13,584, +1.3%, not adopted;
  - `torch.compile(lm.transformer, dynamic=True)`: 14,390 (+7.3%, 1.5 GiB less), but two of its
    frames recompiled per batch shape up to the limit of 64 (warm-up 484 s, not 89).
- The compiled main-transformer forward on the H200 NVL (2026-11-03, the same benchmark):
  14,682 frames/s against 13,838 (+6.1%), 73.1 GiB reserved against 75.1, the same loss.
  Dynamo compiles it twice (batches under 4,096 frames get their own graph, an inductor fusion
  guard), so the 200 warm-up steps took 137 s against 126.
- What is left (warm profile, H200 NVL): matmuls 59% of GPU time, compiled kernels 16%, fused
  AdamW 8% and the grad norm 4% (bandwidth-bound: 87 GB per step), copies 5.5%, eager
  elementwise 3%; the GPU idles 4%, and the loader takes 0.15 ms per step. The first step of
  each new batch shape also builds cuDNN attention plans (~75 ms).
- Generation (`LMGen`, ms per step at batch 16/32/64): fp32 57/93/164, bf16 31/47/80, bf16 with
  GQA 16/18/22.

## By GPU

The same benchmark by GPU (one epoch is 131,553 steps); the H200 NVL and 700 W SXM rows have
the compiled main transformer, worth +6.1% and +7.4% there (13,838 → 14,682 and 13,405 →
14,399), the other two predate it:

| GPU (power limit) | frames/s | s/step | one epoch |
|---|---:|---:|---:|
| H200 NVL (600 W; warm, after 200 steps) | 14,682 | 0.277 | 10.1 h |
| H100 SXM 80 GB (700 W; warm, after 200 steps) | 14,399 | 0.283 | 10.3 h |
| H100 NVL (400 W) | 10,669 | 0.385 | 14.1 h |
| H100 SXM 80 GB (500 W, its host's cap) | 7,401 | 0.550 | 20.1 h |

- The H200 NVL drew 536 W on average at 1,705 MHz (1,785 max) and 58 °C: under its 600 W limit.

- Of four H100 SXM pods rented on 2026-11-03, the three on EPYC 9454 hosts (Germany, Latvia)
  were capped at 500 W; the capped one benchmarked sat at its limit 99% of the time, at a mean
  798 MHz SM clock (1,980 max), with no allocator retries. The uncapped one (Australia) drew
  577 W at 1,887 MHz and 63–68 °C under load. `nvidia-smi -q -d POWER` shows a pod's limit.
- An 80 GB card fits: CUDA sees 79.2 GiB of it and the allocator gets 78.7 GiB. Training
  peaked at 73.6 GiB allocated and 75.1 GiB reserved there before the compiled main transformer,
  and translating FLEURS dev with an 8 s tail at 74.4 GiB; with it, 71.6 and 73.1 GiB on both
  the H200 NVL (where main also reserved 75.1) and the 700 W SXM, with no allocator retries. The
  validation's bf16 generator sits in moshi's reference cycles, so only `gc.collect()` frees its
  5.8 GiB: without it the first training step after a validation ran out of memory there
  (76.8 GiB allocated), which a 141 GB card hides.
- About 33 GiB of that 74.4 GiB was Mimi coding all 64 rows at once in fp32 (0.8 GiB per 40 s
  row, measured on CPU). With the 20 s tail FLEURS dev's biggest batch (64 rows of up to 17 s)
  would have needed about 89 GiB in total, so Mimi now codes 8 rows at a time and validation
  should peak well below training.
- Since 2026-11-03 validation also translates the 360 `en-vi-val` rows (54 minutes of speech)
  in the same batches as FLEURS dev: 12 batches instead of 6, the biggest 64 rows of up to
  22.1 s and the last 17 rows of up to 32 s, at most 4,918 generation steps against 2,647.
- The second run's validation at step 18,000 (H100 SXM, 2026-11-04; five sets, 1,081 rows,
  moshi's full KV cache) took 5.5 min, peaking at 76.1 GB in use (nvidia-smi):
  - the teacher-forced pass took ~110 s with the GPU mostly idle, bound by the pod's 7.7-CPU
    quota, of which an upload took 1.3;
  - generation took 157 s in 17 batches, at 64.6 GB, with CPU gaps between batches.
- From step 36,000 the same validation runs with the batch-sized KV cache (below): 212 s, not
  330 (2026-11-04).
- Training holds 76 GB of the 80, so validation cannot run beside it.
- On an idle H100 with `eval.py`, the KV cache sized to the batch's steps
  ([sft-training](sft-training.md#side-tools)) cut the five sets' generation from 151 to 109 s
  and the peak from 38.5 to 19.5 GiB. Each set's chrF moved by up to 1.3 either way (sampling
  noise); FLEURS went 24.58 → 24.62.
- Teacher-forced validation runs under bf16 autocast, as training's losses do: 50 s for the
  1,429 streams on the capped SXM against 83 s on fp32 weights without it; the loss moves by
  0.01% and PAD accuracy by 1 point (base model). Loader workers save nothing: assembling the
  300 batches takes 0.8 s.
- Translating FLEURS dev (bf16, batch 64, 8 s tail, 1,747 steps): 42 s on the H200 NVL,
  48–53 s on the capped SXM. Batch 96 takes 34 s but 100.3 GiB. Before these changes (fp32,
  batch 16) it took 382 s.
- The checkpoint writer stalls training for the copy to pinned memory (0.6 s per save on the
  H200 NVL) and writes the 37.5 GB in the background (16 s on the H200 NVL pod's disk, 37 s on
  the capped SXM's, both before the fsync). The first save also allocates the pinned buffers:
  8 s and 26 s of stall. With the fsync a pair takes ~40 s on the 700 W SXM's 1.1 GB/s disk, so
  saves closer together than that (20 steps in the end-to-end test) wait for the previous one;
  1,000 steps apart they never do.
- Host RAM of `en_speech_streams_epoch1b` at step 68,750 (700 W SXM, 2026-11-03, one sample):
  the trainer held its 38.5 GiB of pinned checkpoint buffers (shared memory, allocated once) and
  5 GiB of its own; each of the 4 loader workers 3 GiB plus 12 GiB of mapped packs. The cgroup
  counted 147 GiB of its 181 GiB limit, 135 GiB of it reclaimable page cache of the packs, so
  about 50 GiB is the real need, under the 80 GiB `preflight` floor.

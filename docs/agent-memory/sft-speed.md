# SFT training speed

Read when touching `training/finetune/lm_training.py`, or when choosing a GPU pod or measuring
training, validation or checkpoint speed.
Back to [CONTEXT.md](../../CONTEXT.md) · [Fast path](#fast-path) ·
[Measurements](#measurements) · [By GPU](#by-gpu)

The trainer itself is in [sft-training](sft-training.md); the pod procedure in
[docs/finetune.md](../finetune.md).

## Fast path

`lm_training.install()` (run when `common` is imported) replaces four things in moshi 0.2.13
with the same math on the same weights:

- the causal attention mask becomes `is_causal=True` (flash);
- grouped-query attention runs in SDPA (`enable_gqa`) instead of copying each key and value to 16
  heads: in generation that copy is the whole KV cache, twice per layer and step;
- the depformer's training forward runs its 16 codebook steps step-major (`unbind`/`stack`,
  whose backward is one copy, instead of slices whose backward zero-fills a full gradient each),
  compiled with dynamic shapes, with the memory-efficient attention kernel;
- the training gating FFN splits with `unbind` for the same reason;
- each compiled function keeps up to 64 graphs, not dynamo's 8. moshi compiles `_rms_norm`,
  `apply_rope` and `gating_forward_kernel` once per dtype, rank, grad mode, autocast state and
  shape class, and past the limit a function's new variants run eagerly for the rest of the
  process. Training alone builds 12 graphs of `_rms_norm` and reached 8 within its first 11
  steps; its steps then ran within 2% of those at 64. With every helper eager, a step is 6%
  slower and peaks 3 GiB higher (78.2 GiB reserved).

Checked against moshi as shipped (2026-11-03, H100 NVL): fp32 loss equal, logits within 1.7e-6
and every gradient within 3.1e-6; in bf16 both are as far from an fp32 reference (gradients:
median 8.5e-3, max 9.4e-2); fp32 greedy decoding gives the same tokens for 494 steps on 16
FLEURS rows. Sampled decoding diverges from moshi's after a few words (any rounding change
does that), with outputs of the same kind.

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
    frames recompile per batch shape up to the limit of 64 (warm-up 484 s, not 89), cause not
    traced; a gain to take with a training forward for the main transformer, as the depformer
    has. On the H100 NVL it ran at 10,358 against 10,669.
- What is left (warm profile, same SXM): matmuls 49% of GPU time, compiled kernels 11%, copies
  11%, eager elementwise 9.5%, fused AdamW 10% and the grad norm 4% (bandwidth-bound: 87 GB per
  step); the GPU idles 8% in launch gaps under 0.1 ms, and the loader takes 0.15 ms per step.
  The first step of each new batch shape also builds cuDNN attention plans (~75 ms).
- Generation (`LMGen`, ms per step at batch 16/32/64): fp32 57/93/164, bf16 31/47/80, bf16 with
  GQA 16/18/22.

## By GPU

The same benchmark on the final path, by GPU (one epoch is 131,553 steps):

| GPU (power limit) | frames/s | s/step | one epoch |
|---|---:|---:|---:|
| H200 NVL (600 W) | 13,353 | 0.305 | 11.1 h |
| H100 SXM 80 GB (700 W; warm, after 200 steps) | 13,405 | 0.304 | 11.1 h |
| H100 NVL (400 W) | 10,669 | 0.385 | 14.1 h |
| H100 SXM 80 GB (500 W, its host's cap) | 7,401 | 0.550 | 20.1 h |

- Of four H100 SXM pods rented on 2026-11-03, the three on EPYC 9454 hosts (Germany, Latvia)
  were capped at 500 W; the capped one benchmarked sat at its limit 99% of the time, at a mean
  798 MHz SM clock (1,980 max), with no allocator retries. The uncapped one (Australia) drew
  577 W at 1,887 MHz and 63–68 °C under load. `nvidia-smi -q -d POWER` shows a pod's limit.
- An 80 GB card fits: CUDA sees 79.2 GiB of it and the allocator gets 78.7 GiB; training peaks
  at 73.6 GiB allocated, 75.1 GiB reserved, and translating FLEURS dev at 74.4 GiB.
- Teacher-forced validation runs under bf16 autocast, as training's losses do: 50 s for the
  1,429 streams on the capped SXM against 83 s on fp32 weights without it; the loss moves by
  0.01% and PAD accuracy by 1 point (base model). Loader workers save nothing: assembling the
  300 batches takes 0.8 s.
- Translating FLEURS dev (bf16, batch 64, 1,747 steps): 42 s on the H200 NVL, 48–53 s on the
  capped SXM. Batch 96 takes 34 s but 100.3 GiB. Before these changes (fp32, batch 16) it took
  382 s.
- The checkpoint writer stalls training for the copy to pinned memory (0.6 s per save on the
  H200 NVL) and writes the 37.5 GB in the background (16 s on the H200 NVL pod's disk, 37 s on
  the SXM's). The first save also allocates the pinned buffers: 8 s and 26 s of stall.

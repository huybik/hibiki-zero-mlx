# VieNeu TTS throughput optimizations (`training-data/`)

How VI synthetic-speech generation for the PhoMT campaign went from **4.2× to ~53× realtime** on an M4 Pro, and how the "0.45% all-silent wav" bug was root-caused and fixed. All changes live in `training-data/vieneu_mps_patch.py` (applied to the live engine by `pipeline.load_tts`); the vieneu package itself is unmodified.

## The model (vieneu 3.2.3, VieNeu-TTS-v3-Turbo)

Per generated frame: a **Qwen3 semantic backbone** (12 layers, hidden 768, 12 Q / 4 KV heads GQA, head_dim 64) produces one hidden state, then a **2-layer acoustic decoder** autoregressively samples **16 RVQ codebooks** (vocab 1024 each), conditioned codebook-by-codebook. Frames decode to 48 kHz audio through the **MOSS-Audio-Tokenizer-Nano** codec. The model is small, so on MPS the cost is dominated by *dispatch overhead and memory churn*, not FLOPs — that observation drives everything below.

## Phase 1 — batched-engine patch (MPS, 4.2× → ~25×)

`_generate_batch_fast` replaces `V3TurboBatchEngine.generate_batch`:

- **Tensorized repetition penalty**: the stock per-row Python `set` + `.item()` loop caused ~153k GPU→CPU syncs per batch; replaced with a persistent `(B, n_vq, 1024)` bool `seen` mask applied via `torch.where` (CTRL/MOSS rule preserved: `logit<0 → ×p`, else `÷p`, before temperature).
- **Device-side EOS**: `finished`/`lengths` live on device; `finished.all()` is synced only every 8 frames.
- **One batched codec decode** per batch instead of per-row decodes (the codec is causal, so batched results match per-row).
- **fp16 LM** (`VI_FP16`): backbone + acoustic decoder + heads via `model.half()`; codec autocast (upstream helper is CUDA-only, patched for MPS).
- **`infer_batch_voices`**: per-text voices with global batches across all 48 voices (stock groups by voice → tail waste); chunks sorted by phoneme length so batch members finish together.
- **`torch.mps.empty_cache()` per batch**: the MPS caching allocator hoards every freed size class over a long run and swap-thrashes otherwise.
- Kokoro EN moved to CPU workers, freeing the GPU for VI.

## Phase 2 — CUDA path (2026-07-30, not yet run on a CUDA box)

`_GraphedFrameFast` captures the whole per-frame acoustic step (16 decoder steps + sampling + EOS head) as one `torch.cuda.CUDAGraph` replay, keyed per batch size. The rep-penalty survives capture because `seen` is a static device tensor scatter-updated inside the graph (stock vieneu's graph path can't do that). `CUDA_BATCH_SIZE=128` since the graphed step is launch-bound ⇒ ~linear throughput in B. Smoke-test one batch before a full tranche.

## Phase 3 — MPS hyper-opt pass (2026-07-30, ~25× → ~53×)

Profiling (B=32, idle GPU, synced per stage) showed:

| stage | ms/frame | share |
|---|---|---|
| HF Qwen3 backbone decode step | 27 | 51% |
| acoustic frame (16 codebook steps + sampling) | 9.8 | 19% |
| codec batch decode | (per batch) | 18% |
| prefill (per batch) | — | 8% |
| decode-slot embedding | 1.4 | 4% |

### Backbone: `_MPSFastDecoder.bb_step` (27 → 7 ms/frame)

The HF per-step forward paid Python/mask/rope overhead per call, and `DynamicCache` `torch.cat`s — fully re-copies — the KV of all 12 layers every frame. Replaced with a hand-rolled step:

- HF still runs the (once-per-batch, left-padded) **prefill**; its cache is copied into **preallocated static KV buffers** written in place each step. Because prompts are left-padded/right-aligned, all rows write the same cache column; only rope positions differ per row (gathered from a precomputed cos/sin table).
- **Shape bucketing**: MPS compiles a Metal kernel graph per tensor shape. Attention KV length is bucketed to multiples of 64 (extra columns masked), buffer allocations to multiples of 128 — a handful of shapes instead of one per frame/batch.
- **`sdpa(enable_gqa=True)` works on MPS** (torch 2.13) and beat manual grouped-matmul attention ~2.7× per call.
- Exactness: reuses the model's own norm modules and HF's op order. Greedy-decode diff vs the old path: 64/64 codes identical per step, EOS identical, hidden drift ~1e-3 (fp16 reassociation noise).

### Acoustic frame (9.8 → 7 ms/frame) — and an instructive failure

Applying the same static-KV treatment to the tiny 2-layer decoder was **3× slower** (36 ms/frame): at 17-token lengths `torch.cat` costs nothing, while strided writes into a 5-D buffer + unfused manual attention cost a lot. Kept functional-cat KV + fused SDPA, and won instead via:

- precomputed slot position embeddings and the constant SGS text embedding;
- `is_causal=True` for the 2-token prefill, **no mask at all** for single-token steps (a lone new token attends everything — building the additive bias each step was pure waste);
- **sort-free sampling** (2.94 → 1.22 ms): rep-penalty on full logits → `topk(25)` → softmax + nucleus mask *within candidates* (identical support/probabilities) → **exponential-race sampling** `argmax(probs / Exp(1))` ≡ `multinomial(probs, 1)` → gather. No full-vocab sort, no `multinomial`.

### Decode-slot embedding (fused)

Stock `_build_inputs_embeds` did 16 embedding lookups + pad masks per frame (~80 launches). Generated codes are always valid, so it collapses to: one gather over a stacked `(16·1024, 768)` embedding table + sum + a precomputed constant (SGS text emb + speaker anchor). ~4 launches.

### Measured, and things that didn't pay

- **~53× RT warm at B=32** (uniform-length bench); **52.8× sustained** over 256 random real rows; live tranche batches of 32 completing every 2–3 s.
- **B=64: no gain** — wall time ≈ sum of GPU stages, the GPU is saturated; codec is now the largest stage (~28%).
- `torch.compile`: nothing left worth fusing after the SDPA swap; skipped.
- Per-voice prefix-KV caching of the reference-codes prompt segment: ~7% upside, real complexity; not done.
- `MPS_FAST = False` module flag = A/B kill-switch back to the old loop.

## Silent-wav root cause and fix (2026-07-30)

The historical ~0.45% all-silent VI wavs were **NaNs born inside the MOSS codec**, written to PCM as silence:

1. **Padding NaN**: in a mixed-length batched decode, short rows are right-padded; their padded positions get *fully-masked* fp16 attention rows inside the codec's transformer → NaN softmax → NaN spreads over the entire row (global attention). This is why regenerating stragglers at `batch_size=4` "fixed" them — less padding. Reproduced on the pre-optimization path too (it was never a fast-path regression).
2. **Rare content-driven fp16 overflow** in the codec convs (~1/64 batches even with uniform lengths).

Fix, both in the patch: every row in a batched codec decode now has the **same length** — short rows padded by repeating their own last valid frame (causal codec ⇒ the real prefix decodes identically; the tail is trimmed by `lens[b] × samples_per_frame`) — and the codec autocast runs in **bf16** (fp32 exponent range kills the overflow; LM stays fp16, it was never the source). Validation: 256 rows → **0 NaN, 0 silent**, duration ratio vs old output median 1.000. The tranche silence scan is now a safety gate; wavs generated *before* this fix still need the full scan.

## Kokoro EN CPU pass (2026-07-30)

Profiling showed the iSTFTNet decoder is **92%** of Kokoro time (G2P negligible). Key finding: aggregate EN throughput on the M4 Pro is **memory-bandwidth-bound at ~22× RT** — 10w×1t, 13w×1t and 7w×2t all land within 6%, so worker/thread shuffling can't push past the ceiling; only cutting memory traffic per sample does. What worked: fold `weight_norm` (frees compile from tracing the norm hook) + `torch.compile(decoder, dynamic=True)` → **25.8× aggregate at 10 workers × 1 thread** (`EN_COMPILE` in `pipeline.py`; warmup ~50 s/worker warm-cache, ~8 min cold — `TORCHINDUCTOR_CACHE_DIR` pinned to `~/.cache/torchinductor-kokoro` so it survives reboots). Dead ends, all measured: ONNX Runtime is 3× *slower* than torch eager on this graph (all variants ≈1.8× RT, quantized included, fp16 NaNs); CPU bf16/fp16 convs fall back to a reference kernel ~186× slower than fp32; thread scaling tops out at 2.2× from 1→8 threads. Validation: 24 manifest rows regenerated — 0 NaN/silent, duration ratio 1.000. Next EN lever if ever needed: CoreML/ANE for the decoder (the ANE is idle; dynamic shapes make it nontrivial).

## Kokoro EN Core ML / Metal GPU pass (2026-07-30, ~26× → ~70× aggregate)

The iSTFTNet decoder (94% of EN wall) now runs as a Core ML mlprogram on the Metal GPU (`training-data/kokoro_coreml.py`): 73 ms vs 927 ms compiled-CPU for 6 s of audio (12.7×). **ANE measured and rejected**: the E5 compiler ground 26 min on the graph and emitted a program no faster than eager CPU. Design points, in dependency order:

- **Split**: the random sine source + har STFT + 20-point iSTFT stay in torch on CPU (<3% of time; random/complex ops don't convert). The fp32 split is bit-exact vs the stock decoder. har is computed from the *unpadded* F0 (then zero-padded), which keeps its tail STFT frames and its RNG draws identical to the CPU path.
- **RangeDim + 32-frame buckets**: flexible shapes cost nothing on the GPU, but every first-seen shape pays ~600 ms of Metal specialization — so T is zero-padded to a multiple of 32 (~20 warm shapes per run).
- **Artifact boundary**: the cache key covers the conversion schema, dependency versions, and exact decoder state. A process lock serializes cold conversion, which publishes through a same-filesystem atomic rename so workers only see complete packages; each worker then opens its own `MLModel`.
- **Exact masked InstanceNorm**: AdaIN normalizes over time, so naive padding shifts every chunk's stats (corr ~0.9). The graph is traced with masked stats (1/n-prescaled prefix masks as inputs — summands stay at mean/variance scale, so everything survives fp16 without fp32 pinning) **plus pad-tail re-zeroing at every AdaIN output, around the transposed convs, on har, and before conv_post** — each conv then sees exactly the zeros the unpadded graph's implicit conv padding provides. Torch A/B: corr 1.000000; Core ML end-to-end: corr 0.99993–0.99999 across all 7 voices (pure fp16 rounding).
- **eps 1e-4, not 1e-5**: 1e-5 is subnormal in fp16 and Metal flushes it, so a near-constant channel on a long chunk hits rsqrt(0)=inf → NaN.
- **Metal buffer-reuse bug + CPU rescue**: specific long-chunk shape/value sequences (portable repro: T 1184→800→480→1088 with real values; synthetic values or any subsequence are clean) poison the *process-wide* GPU context for that feed, deterministically — even a fresh MLModel instance fails, but CPU execution is immune. The corruption manifests as **NaN or as finite near-silence** (chunk rms ~0.011 vs ≥0.036 on every healthy chunk, n=200), so `CoreMLDecoder` gates every chunk on finite + rms ≥ 0.02 and reruns implausible ones on a CPU_ONLY instance (~2.5 s, extreme-length clips only). 48-row production gate (incl. 49.75 s outlier): PASS 3/3, duration ratio 1.0000.
- **Layout**: macOS uses 7 all-CoreML EN workers and saturates the GPU at **~70–75× RT aggregate** (16.4×/worker warm; near-linear to 4 workers). Other platforms retain the 10-worker compiled-CPU layout.
- **Contention with VI is negligible**: EN 65.6× (vs 68.5× solo) while VI runs at ~90% of solo — vieneu's launch-bound MPS kernels and Core ML's predicts interleave. The tranche long pole flips from EN (26×) to VI (~53×): ~2× faster tranches.

## Concurrent-pipeline profiling: the GPU-saturation ceiling (2026-07-30, E1–E14)

Instrumented production-shaped runs (512 rows each, sync-fenced stage timers; solo bench numbers above do NOT survive concurrency):

- **Baseline concurrent (1 VI + 7 EN): VI 33×, EN ~40–50× — the M4 Pro GPU is fully allocated.** EN contention costs VI ~40% (33× vs 55× solo), not the ~10% previously believed. A 96k-row tranche ≈ 5.3 h (was 17 h on the pre-optimization overnight run).
- **Scheduling-level ideas are dead ends, all measured:** a second VI worker adds nothing (34× combined, +6 GB swap); EN worker count barely matters (5 ≈ 7 for both languages); so rebalancing shuffles who waits, not throughput.
- **Codec placement, all measured:** CPU-offloaded codec (dedicated subprocess; threads+GIL version even worse) reaches only ~40× RT on real length-sorted batches (padding waste + O(T²) attention; 82× on the uniform microbench) and ~28× under the EN workers' CPU load — below the LM pace it must hide behind, a net loss (solo 34×, concurrent 23×). In-process **async GPU decode crashes**: torch MPS forbids command encoding from two threads (`MTLCommandBuffer` assertion). GPU-synchronous is the optimum.
- **Free-threaded CPython 3.14 re-check (repo `.venv314t`, GIL off, torch 2.13 cp314t): the GIL was never the limiter.** On a 260-chunk preset-voice solo workload, py3.12-with-GIL and 3.14t produce identical numbers in every variant: GPU-sync baseline 72×, LM-only (codec skipped) ~110×, LM+1 CPU-codec thread pipelined 50× (LM pace barely dips, ~99×, in *both* — torch CPU kernels release the GIL), 2 codec threads 63×. The limiter is the CPU codec itself (~56× RT one thread, ~40×/thread with two sharing the intra-op pool), below the LM pace it must hide behind. Codec stays GPU-synchronous. (This bench's solo baseline runs 72× vs the 53× figure above — presets-only prompts and a different row range; comparisons are within-bench.) Env notes: uv venv `3.14t`; vieneu installed `--no-deps` (onnxruntime has no cp314 macOS wheel, sea-g2p's pyo3 0.23 can't build free-threaded — both stubbed in the bench, phonemes pre-dumped from the py3.12 env); tokenizers pinned `0.23.0rc0` (0.22.x re-enables the GIL on import); Rust deps built with `RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"`.
- **Shared codec attention mask** (kept): our decodes are always uniform-length, so the per-row `(B,1,S,S)` bool SDPA masks collapse to one broadcast `(1,1,S,S)` row — ~330 MB less per batch at T=100, −26% codec on uniform microbench, ~−8% on real batches, end-to-end within noise but free memory relief.

## Silent-wav mechanism #2: all-zero Metal rows (2026-07-30, found by full-tranche scan)

Scanning all 21k VI rows generated tonight found **0.44% all-zero wavs on the stock path** (232 total; biased to long rows, up to ~4% of 8 s+ rows; every voice affected; EN clean). Signature: full-length, exactly-zero PCM — Metal silently returns zero rows for some long-T decodes (stage-7 seq = 32T ≈ 4–10k, multi-GB mask/workspace pressure), the same failure class as the Kokoro Core ML buffer bug. This is a **new mechanism, not the historical NaN one**: the 96k-row 249600–345599 tranche (generated on the pre-Phase-3-day stack) scanned fully clean, so the corruption arrived with today's higher GPU memory pressure (Phase-3 static-KV buffers + 7 Core ML EN workers co-resident). CPU-codec rows and (small-n) shared-mask rows showed none. Fix in `_generate_batch_fast`: every decoded row is gated on finite + max|x| > 0; corrupt rows re-decode on a lazily-created **CPU clone** of the codec (immune, ~40× RT, rare) and the run raises if a row fails both. Tonight's 232 rows were regenerated and verified.

## Current balance & next levers

Concurrent VI 33× + EN ~40–50× is the measured GPU-saturation ceiling on the M4 Pro; only removing GPU work per audio-second can move it. Untested ideas, in value order: MOSS codec as Core ML on the **ANE** (the only idle silicon; transformer stack is ANE-friendly unlike iSTFTNet, but the O(T²) sliding-window masks need banded attention or fixed-T buckets, and ANE fp16 reopens the overflow question); EN batched predict (amortize the ~250 ms/chunk dispatch; masked-norm design already supports per-row masks); per-voice prefix-KV prefill cache (~10% of VI GPU).

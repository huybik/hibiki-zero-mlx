# Grounded & Noisy Cache Generation Optimization Plan (RTX 5090 / Blackwell)

Date: 2026-11-28

Systems, kernel, Blackwell (`sm_120`) FP16/FP8, TensorRT, and CPU/IPC optimization plan for speeding up grounded cache generation (`training/prep/cache_build.py`, `cache_rows.py`, `text_timing.py`, `cache_codes.py`, `training/tranche/cache.py`) and noisy source re-encoding (`training/prep/noisy_cache.py`, `augment.py`) on a single 32 GB RTX 5090 (`sm_120`, 575 W TDP, 1.79 TB/s GDDR7).

Related notes: [cache-build](agent-memory/cache-build.md) · [cache-speed-cuda](agent-memory/cache-speed-cuda.md) · [tranche-cache](agent-memory/tranche-cache.md) · [noisy-source](agent-memory/noisy-source.md) · [envi-data](agent-memory/envi-data.md) · [tranche-yodas](agent-memory/tranche-yodas.md) · [sft-speed](agent-memory/sft-speed.md).

---

## 1. Current Baseline & Diagnosis

### 1.1 Measured RTX 5090 Baseline

Measured on box `61906288` (RTX 5090 32 GB, driver 580, 15.4-CPU quota, 31 GB RAM, 15 GB `/dev/shm`; logs in `/Volumes/data/hibiki-zero-evals/box-61906288/envi_box/prof/`) and box `62513802` (RTX 5090 32 GB, driver `595.91.07`, 19.2-CPU quota):

| Workload | Measured 5090 Speed | Active GPU Util & Power | Peak VRAM / Host RAM | Cost for 20k h Scale-Up (`vi-en` + `en-vi` + noisy) |
|---|---:|---:|---:|---:|
| **Grounded build (`en-vi` / `vi-en`)** (`cache_codes` / `tranche.cache`, `window=512`, `CUDA_BATCH_S=320`, TF32, 3.14t no-Triton) | **~2,130–2,200× RT** steady (`1,676×` on 1 shard w/ 14 s start) | 97–99% SM busy, **480 W avg** (p50 490 W, max 508 W vs 575 W TDP) | 16–30 GiB VRAM / 10.4–13.2 GB RAM | `~39.0 h` for both directions (`~84k h` Mimi audio) |
| **Noisy re-encode (`noisy_cache encode`)** (`--batch-seconds 320`, TF32, Mimi-only on 60% of rows) | **~2,780–3,182× RT** steady (`~2,130×` w/ startup/save on 1 shard) | 91.7% active SM (**only 30% of 1-shard wall active**), **449 W avg** | 27.0 GiB VRAM / 4.7–18 GB RAM | `~4.3 h` for `en-vi` noisy (`~12k h` re-encoded) |
| **Combined total** | — | — | — | **~43.3 5090-h (~$22.10 @ $0.511/h)** |

*(On the 1,000 h YODAS pilot on box `62513802`: `vi-en` ran `2,129× RT` and `en-vi` ran `2,180× RT`, taking `44 min` each way = `88 min` total.)*

### 1.2 Exact GPU Time, FLOP, and HBM Bandwidth Breakdown (Per `320 s` Padded Batch)

From Mimi-only (`~3,182× RT` = **`~100.5 ms` per 320 s of Mimi audio**) vs full build (`~2,180× RT` = **`~146.8 ms` per 320 s of Mimi audio**, which includes `~152 s` of 16 kHz CTC audio since Mimi encodes both clips + delay ≈ `2.1×` target length):

| Model Stage | Share of Build GPU Time | Est. Time / 320 s Batch | Exact FLOPs / 320 s | Effective TFLOP/s (vs 5090 Peak) | Primary Bottleneck |
|---|---:|---:|---:|---:|---|
| **Mimi SEANet (24 kHz → 25 Hz)** | **~50%** | ~73 ms (per 320 s Mimi) | 1.30 TFLOP | **~18 TFLOP/s** (~8.5% of TF32) | **~60+ GB fp32 HBM traffic** per 320 s batch! Stage 0 tensor (`7.68M × 64` fp32) is **1.97 GB**; `StreamingConv1d` `torch.cat` zero-pads + eager `ELU` + residual add copy it **~17×**! |
| **Mimi Transformer (8L, `d=512`, `ctx=250`)** | **~11%** | ~16 ms (per 320 s Mimi) | 0.455 TFLOP | ~28 TFLOP/s | Eager `apply_rope` (`NO_TORCH_COMPILE=1`) + 8× rebuilt `(B, 1, T, T)` boolean sliding-window mask in fp32 non-Flash SDPA |
| **Mimi RVQ (16 codebooks, `d=256, K=2048`)** | **~7%** | ~11 ms (per 320 s Mimi) | 0.067 TFLOP | ~6 TFLOP/s | 16× `b c t ↔ b t c` transposes + `torch.cdist` recomputing `‖x‖²`, `‖e‖²`, `sqrt` instead of `half_norm - r @ e.T` |
| *Mimi subtotal* | ***~68%*** | *~100 ms (per 320 s Mimi)* | *1.82 TFLOP* | ***18.2 TFLOP/s*** | *Heavily memory-bandwidth & launch bound in fp32* |
| **wav2vec2 Conv FE (`c0..c6`, `16 kHz → 50 Hz`)** | **~11%** | ~34 ms (per 320 s CTC) | 1.57 TFLOP | ~46 TFLOP/s | `c0` output is **2.10 GB** fp32; manual masked GroupNorm (`cache_build.py:422-427`) makes **~10 HBM passes (~20 GB = ~13 ms)** alone! |
| **wav2vec2 12L Transformer (`d=768, ff=3072`)** | **~21%** | ~63 ms (per 320 s CTC) | 3.19 TFLOP | ~51 TFLOP/s | Uncompiled eager fp32/TF32; unfused `q/k/v_proj`; `weight_norm` recomputed every batch on `pos_conv_embed`; fp32 non-Flash SDPA (17%), copies (15%), LN (7%), GELU (6%) |
| *wav2vec2 CTC subtotal* | ***~32%*** | *~97 ms / 320 s CTC (~46 ms / 320 s Mimi)* | *4.76 TFLOP* | ***49.1 TFLOP/s*** | *Eager fp32/TF32; never tested FP16 (10-bit mantissa) or Triton/compile on CUDA* |

### 1.3 Critical Discovery — The Hidden CPU Ceiling (`build_w14.log`)

From `/usr/bin/time -v` on `libriheavy-large-00012-00103` (`24.93 h` = `89,748 s` of Mimi audio, `3,147` rows, `box-61906288/envi_box/prof/build_w14.log`):

- **Total CPU time:** `196.32 s user + 37.02 s sys = 233.34 CPU-s` (`~74 CPU-ms/row` = **`385× RT` per CPU core**).
- **Hard CPU throughput ceiling today:**
  - 15.4-CPU box (`61906288`): `385 × 15.4 ≈` **`5,900× RT`**.
  - 19.2-CPU box (`62513802`): `385 × 19.2 ≈` **`7,400× RT`**.
- **Kernel syscall tax:** **`37.02 s` sys CPU (16% of all CPU!)**, `7,760,576` minor page faults, and `~2M` FS ops on a **single shard** — caused by creating, writing, reading, and unlinking **one `/dev/shm/*.f32` file per row** (`cache_rows.py:247-251`) plus copying silence-padded arrays.
- **Implication:** Today CPU sits at ~35% because the GPU only runs at `2,180× RT`. Once GPU kernels speed up `3–5×` (`8,000–13,000× RT`), **the CPU side will immediately bottleneck the RTX 5090** unless we optimize both GPU and CPU/IPC together.

### 1.4 Why Triton & `torch.compile` Were Disabled (And Why That Was a Local Optimum)

`envs-min.sh:19-27` and `restart.sh:43` run `/workspace/envs/cache` on free-threaded Python 3.14t with **`triton` uninstalled** and **`NO_TORCH_COMPILE=1`** solely because `CUDAModels` runs in a **thread** inside the main process (`Builder._stream`, `cache_build.py:533`), where Triton re-enabled the GIL and caused ~1.8% contention (`3,409× → 3,471×` on H100).

Moving the GPU runner into a **dedicated `spawn` subprocess** eliminates GIL contention completely while unlocking **Triton, `torch.compile`, CUDA Graphs, TensorRT 10.x, FlashAttention-2, and Blackwell FP8 (`torch._scaled_mm`)** — worth **`3–5×`**, not `1.8%`.

---

## 2. Part A — Optimizing Mimi Encoding on RTX 5090 (`100 ms → ~16–22 ms` / 320 s)

Mimi accounts for **~68% of grounded-build GPU time** and **~90% of `noisy_cache` GPU time**.

### A0. Crucial Precision Finding: FP16 Storage + FP32 Accumulation (Same 10-Bit Mantissa as TF32!)

Earlier experiments (`cache-build.md:150`, `cache-speed-cuda.md:97`) rejected:
- **bf16 Mimi on CUDA** (only **7 mantissa bits** → rewrote 55% of residual codes), and
- **global fp16 Mimi on MLX** (ran `torch.cdist` L2 subtraction and `layer_scale = 0.01` in fp16 with fp16 accum → 89% equal).

**Why FP16 + FP32 accumulation on RTX 5090 is fundamentally different:**
- **Production TF32** has **10 mantissa bits** (`99.92%` cb0 equal, `96–99%` residual equal, decode SNR `9.043` vs `9.040 dB` fp32).
- **FP16** has the **exact same 10 mantissa bits** (`eps = 9.8e-4`) and covers `[6e-5, 65504]`, well above SEANet's activation range (raw audio `[-1, 1]`, no LayerNorm).
- Running **SEANet in FP16 I/O with FP32 Tensor Core accumulation** (`FP16 × FP16 → FP32`), while keeping **Transformer residuals/LayerNorm and RVQ L2 quantization in FP32** (where `T = 4k–8k`, <100 MB HBM):
  - **Halves SEANet HBM traffic** (Stage 0 tensor `1.97 GB → 0.98 GB`),
  - **Halves SEANet activation VRAM** (eliminates the 27–30 GiB VRAM pressure on 32 GB 5090),
  - **Doubles Blackwell Conv/GEMM Tensor Core throughput** vs TF32,
  - **Unlocks FlashAttention-2 / cuDNN SDPA** in Mimi's 8-layer encoder transformer,
  - **Expected parity:** Near-identical to production TF32 (`≥99.9%` cb0, `~96–99%` all 16, decode SNR within `±0.01 dB` of fp32). Verify on 256 clips before adopting.

### A1. Stateless Channels-Last (`(B, T, C)`) Encode-Only CUDA Mimi (`training/prep/mimi_cuda.py`)

Replace `moshi`'s streaming `MimiModel.encode()` with a clean, stateless, encode-only CUDA module (mirroring `training/prep/mimi_mlx.py`, ~180 lines loading the exact same `mimi-pytorch-e351c8d8@125.safetensors` weights):

1. **Eliminate `StreamingConv1d` padding copies (`moshi/modules/conv.py:245-274`):**
   - Today every conv allocates `_StreamingConv1dState`, calls `torch.cat([zeros, x], dim=-1)` (copying up to `1.97 GB` just to prepend 2–8 zeros!), and runs `torch.where` state updates even when not streaming.
   - A stateless encoder does zero state allocations and zero `torch.cat` copies.
2. **Fused Triton SEANet Residual Blocks (Stages 0–2: `C = 64, 128, 256`):**
   - Port the proven Metal kernel from `mimi_mlx.py:50-107` to **Triton** on Blackwell (`sm_120`):
     $$\text{y} = \text{ELU}\big(x + W_2\,\text{ELU}(\text{conv}_{k=3}(\text{ELU}(x)) + b_1) + b_2\big)$$
   - Keeps `H = 32` (Stage 0), `64` (Stage 1), `128` (Stage 2) entirely in SRAM/registers; causal left padding (`t - 2 + k < 0`) is a single bounds mask.
   - Fuses the trailing `ELU` (input to the downsample conv) directly into the store.
   - **HBM traffic reduction:** Stage 0–2 residual + ELU traffic drops from **`~43.4 GB` (fp32 unfused) → `~3.4 GB` (fp16 fused) — a `12.8×` reduction in HBM bytes!**
3. **Zero-Copy Strided Downsample GEMMs (`k = 2s`, `s ∈ {4, 5, 6, 8}`):**
   - Exactly as in `mimi_mlx.py:131-139`, every SEANet downsample has `kernel_size = 2 * stride`:
     - In `(B, T, C)` layout, `x.view(B, T // s, s * C)` is a **zero-copy view**.
     - Pre-pack weight into `W_both` of shape `(s * C, 2 * C_out)` at load time.
     - Run **1 pure cuBLAS GEMM** (`M = 1.92M / 384k / 64k / 8k`, all dimensions multiples of 64 — peak Blackwell Tensor Core shapes!) + 1 fused Triton `shift_add_bias` kernel (`out[t] = y[t-1, :C_out] + y[t, C_out:] + bias`).
     - Zero `torch.cat` padding copies, zero NCL↔NLC transposes.
4. **Fast Mimi Encoder Transformer (`8L, d=512, 8 heads, ctx=250`):**
   - Precompute RoPE `(cos, sin)` tables once up to `T = 600` (eliminates 8× eager `arange/exp/cos/sin/stack` in `moshi/modules/rope.py:45-80`).
   - Run compiled FP16 attention with **FlashAttention-2 (`flash_attn_func(..., causal=True, window_size=(249, 0))`)** or PyTorch 2.15 `flex_attention` / cuDNN SDPA (eliminates 8× dense `(B, 1, T, T)` boolean mask allocations in `moshi/modules/transformer.py:575-585`).
   - Compile the 8-layer block (`@torch.compile(dynamic=True)`) to fuse `LayerNorm`, `LayerScale`, `GELU`, and residual adds (keeping LayerNorm and residual accumulations in fp32).
   - *(Phase 2 quality-gated)*: test pre-quantized FP8 (`FrozenFP8Linear`) on the 8 layers' `in_proj`, `out_proj`, `linear1`, `linear2` (`0.40 TFLOP / 320 s`).
5. **Transposeless Precomputed-Norm RVQ (`16` codebooks, `d=256, K=2048`):**
   - Stay in `(N, 256)` (`N = B * T = 4,000` at 320 s) — eliminates 32 `rearrange` transpose copies (`core_vq.py:399-405`).
   - Precompute `emb[k]` (`[2048, 256]` fp32) and `half_norm[k] = 0.5 * (emb[k]**2).sum(-1)` once at load (`mimi_mlx.py:224`).
   - Per codebook: `dot = residual @ emb[k].T` (1 TF32/FP32 GEMM) → fused Triton `(half_norm - dot).argmin()` + `residual -= emb[k][idx]` (avoids materializing `[4000, 2048]` `32 MB × 16 = 512 MB` to HBM and skips `‖r‖²` and `sqrt`).

### A2. Alternative / Benchmark: TensorRT 10.x FP16 Engine for Mimi SEANet + Transformer

- Export the stateless `MimiSEANet + EncoderTransformer + Downsample` (stopping right before RVQ) to ONNX and compile a **TensorRT 10.x FP16** `.plan` engine with dynamic profile `(B ∈ [1, 32, 64], T ∈ [1920, 384000, 537600])` (or flattened/bucketed `T` multiples of `1920`).
- TRT 10.x Myelin automatically fuses `Conv1d + ELU + Residual Add` in NHWC with Blackwell cuDNN/CUTLASS kernels and builds once in ~30 s (<1 s load).
- **Benchmark A1 (PyTorch + Triton) vs A2 (TensorRT 10.x FP16)** on the same 5090 shards and keep whichever is faster with equal decode SNR (`≥9.04 dB`).

**Expected Mimi impact:** **`~100 ms → ~18–24 ms`** per `320 s` batch (**`~4.2–5.5×` speedup**, `~13,000–17,800× RT` Mimi alone; VRAM `27–30 GiB → ~6–9 GiB`).

---

## 3. Part B — Optimizing wav2vec2-base CTC on RTX 5090 (`97 ms → ~15–22 ms` / 320 s)

Both aligners (`facebook/wav2vec2-base-960h` and `nguyenvulebinh/wav2vec2-base-vietnamese-250h`) share the exact `wav2vec2-base` architecture (`95M` params: 7 Conv1d layers + 12 post-norm Transformer layers, `d=768, 12 heads, ff=3072`).

### B1. FP16 Compute with FP32 GroupNorm & `lm_head` (Quality-Proven!)

- **Why earlier `--half-ctc` failed on CUDA:** It used **`bfloat16`** (7 mantissa bits → 96.9% min per-clip argmax, failed the ≥99% gate) and ran slower than TF32 on H100 (`cache-build.md:166-168`).
- **Why FP16 succeeds:** `cache-build.md:132` already proved on Metal that **FP16** (10 mantissa bits, same as TF32) with fp32 GroupNorm stats keeps **99.965% of argmaxes on Vietnamese** and **99.86% on English (min clip 99.24% — passes the ≥99% gate!)**.
- On RTX 5090, run `c1..c6`, `feature_projection`, `pos_conv_embed`, and the 12 Transformer layers in **FP16** (`torch.float16`, FP32 accumulation), keeping `first.conv`'s GroupNorm statistics and `lm_head` + `log_softmax` in **FP32**.

### B2. Fused Triton Masked GroupNorm + GELU on `c0` (`20 GB → 2.1 GB` HBM!)

Target: `training/prep/cache_build.py:420-428`.

- `first.conv(batch)` outputs `(B, 512, T/5)` = **`2.10 GB` in fp32** (`1.05 GB` in fp16) at `320 s`.
- Currently lines 422–428 launch 6 eager ops (`_within`, `hs * own`, `(hs - mean).square() * own`, norm, affine, `GELU`) moving **`~20 GB`** through GDDR7 (**~13 ms/batch**).
- Write a **two-pass Triton kernel** (`triton_masked_groupnorm_gelu(hs, valid_lens, weight, bias, eps)`):
  - Grid `(B, 512)` — one CTA per `(clip, channel)`, computing fp32 Welford/sum+sumsq over `0..valid[b]` in registers, then normalizing, applying affine + `GELU`, and writing **FP16** directly (zeroing `>= valid[b]` in-place).
  - Drops HBM traffic **`20 GB → 2.1–3.1 GB`** (**6–9× less bandwidth**, saves **~11 ms/batch**!).

### B3. Packed Varlen FlashAttention-2 + Fused QKV + Compiled 12L Encoder

Target: `training/prep/cache_build.py:383-401`, `Wav2Vec2Encoder`.

1. **Fold `weight_norm` once at load:** `pos_conv_embed.conv` (`modeling_wav2vec2.py:358`) recomputes `g * v / ‖v‖` on every batch. Call `torch.nn.utils.parametrize.remove_parametrizations(pos_conv_embed.conv, "weight")` (or `remove_weight_norm`) once in `__init__`.
2. **Fuse `q_proj`, `k_proj`, `v_proj` into one `qkv_proj (768 → 2304)`** per layer at load time (cuts 36 GEMMs → 12 and reads `hidden_states` once instead of 3×).
3. **Pack valid frames + Varlen FlashAttention-2 (`flash_attn_varlen_func`):**
   - Right after `pos_conv_embed` + `layer_norm`, compact `[B, T_max, 768]` to `[total_valid_frames, 768]` with `cu_seqlens` (computed from `frames` on GPU).
   - Run all 12 Transformer layers on packed tokens using `flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=False)`:
     - **Eliminates 100% of attention padding**,
     - **Eliminates the `(B, 1, T, T)` bidirectional mask** (`create_bidirectional_mask`),
     - **Replaces slow fp32 memory-efficient attention** (17% of CTC kernels) with native Blackwell FP16 FlashAttention-2.
   - Run `lm_head` + `log_softmax` on packed valid tokens (`[total_valid_frames, V]`) and transfer **one flat contiguous packed buffer** to pinned host memory — eliminates padded output D2H transfer too!
4. **`torch.compile` the Feature Extractor (`c1..c6`) & 12L Encoder** (now unlocked by the dedicated GPU process) — fuses `LayerNorm`, residual adds, and `GELU` (eliminating the ~28% copy/LN/GELU overhead measured in `cache-speed-cuda.md:123-124`). Alternatively, benchmark a **TensorRT 10.x FP16** engine for `c1..c6` + `pos_conv_embed` + 12L encoder.

### B4. Blackwell FP8 `FrozenFP8Linear` for wav2vec2's 12 Transformer Layers (Phase 2, Quality-Gated)

- The 12 Transformer layers' linears (`qkv`, `out_proj`, `fc1`, `fc2`) represent **`2.72 TFLOP` = 57% of all wav2vec2 FLOPs**.
- Replacing them with pre-quantized rowwise E4M3 `FrozenFP8Linear` (`training/finetune/fp8_linear.py:310-329`, already proven at **650–735 TFLOP/s** on RTX 5090 — `~3.3×` faster than TF32, `~1.7×` faster than FP16) drops the 12L linear compute from `~8–9 ms` to `~4–5 ms` per 320 s batch.
- **Strict verification gate:** Must achieve **≥99.0% per-clip argmax agreement** (on every clip, both EN and VI), identical accept/reject decisions, and `<0.1%` text token frame shifts on a full Libriheavy and YODAS shard before enabling.

**Expected wav2vec2 CTC impact:** **`~97 ms → ~18–22 ms`** (Phase 1 FP16+Triton+varlen FA2) or **`~13–16 ms`** (Phase 2 + FP8) per `320 s` of CTC audio (`~6.5–11 ms` per `320 s` of Mimi audio — **`~4.5–6.5×` speedup**!).

---

## 4. Part C — GPU Pipeline, Overlap & Host Staging (`cache_build.py`)

Once Mimi drops to `~20 ms` and CTC to `~10 ms` per 320 s of Mimi audio, fixed per-batch and per-window overheads dominate unless removed:

1. **Dedicated GPU Subprocess (`multiprocessing.spawn`) + Re-enable Triton (`envs-min.sh`):**
   - Run `CUDAModels` inside its own `spawn` GPU worker process on `torch==2.15.0+cu134` with **Triton, `flash-attn` / `flashinfer`, and `tensorrt` installed** (matching `.venv` instead of the crippled Triton-less `3.14t` cache env).
   - Main process + decode/align workers communicate via shared memory + lightweight queues; zero GIL contention with the GPU launcher.
2. **Preallocated Double-Buffered Pinned Output Staging (`cache_build.py:341-349`):**
   - Currently `_to_host()` calls `torch.empty(tensor.shape, ..., pin_memory=True)` on **every single CTC and Mimi batch** (~60 pinned allocations per window across varying shapes).
   - Replace with 2 preallocated flat pinned output buffers per model (matching `_pinned` for inputs), sliced by `.numel()`.
3. **2-Stream / MPS Overlap of CTC (Compute-Heavy) and Mimi (Memory-Heavy) + Larger / Continuous Windows:**
   - Today `CUDAModels.encode` drains the entire GPU queue twice per 512-row window (end of `ctc_log_probs`, end of `mimi_codes`), and `gpu_w14.csv` shows the 5090 averaging `480 W` vs `575 W` TDP.
   - Replace barrier-synchronized 512-row windows with **continuous micro-batch queues** (e.g., dispatch Mimi batches on `stream_mimi` concurrently with CTC batches on `stream_ctc` — or run **2 MPS cache worker processes** on the 32 GB 5090 since FP16+fused Mimi+CTC uses only **~6–8 GiB** per worker!).
   - With 2 MPS workers (each taking disjoint shards, or splitting `vi-en` and `en-vi`), one worker's compute-bound wav2vec2 Transformer overlaps the other's memory-bound SEANet + CPU staging, pushing the 5090 close to 575 W and hiding all shard-boundary / startup gaps!
4. **Move Delay Silence Padding & `24 kHz → 16 kHz` Resample + Normalization onto GPU:**
   - Currently `cache_rows.py:132-135, 178` resamples `target → 16 kHz`, normalizes `target16`, and calls `np.concatenate([np.zeros(pad), target24])` on CPU, then writes all three (including all the delay zeros!) to `/dev/shm` and DMAs them over PCIe.
   - Instead:
     - Pass only raw undelayed `source24` and `target24` + integer `pad_samples` from workers → **eliminates delay zero-fill, `/dev/shm` I/O, and PCIe transfer of ~8% zeros** (just place `target24` at offset `pad_samples` in `_stage` or on GPU!).
     - Compute `target16` on GPU from `target24` via batched `torchaudio.transforms.Resample(24000, 16000)` (`2:3` exact polyphase FIR `F.conv1d` on CUDA, `<0.05 ms/batch`) + GPU zero-mean/unit-var normalization!
       - **Note:** Unlike Mimi (whose residual codebooks are sensitive to `soxr` vs FFT), wav2vec2-base operates on learned 16 kHz conv features (`50 Hz`) and is insensitive to Kaiser-windowed FIR vs `soxr` — verify ≥99.9% argmax match on 1 shard; if verified, this **cuts `soxr` CPU calls by 33%** and **`/dev/shm` + host RAM traffic by ~25%**!

---

## 5. Part D — Breaking the CPU, IPC & I/O Ceiling (`5,900–7,400× → 18,000+× RT`)

Without Part D, the `233 CPU-s / 24.93 h` CPU cost measured in `build_w14.log` caps a 16–19 CPU RTX 5090 box at **`~6,000–7,400× RT`** regardless of how fast the GPU runs.

### D1. Dual-Direction Single-Pass Build (`--direction both`)

Target: `training/tranche/cache.py`, `training/prep/cache_build.py`, `training/tranche/box/restart.sh:43-46`.

- Every new corpus (e.g., 20,000 h YODAS) is cached **both ways** (`vi-en` and `en-vi`). Running `tranche.cache` twice reads all 4 parquets twice, runs `gate()` twice, and decodes + resamples (`soxr 24 kHz`) every Opus clip **twice**.
- A `--direction both` mode (loading both English and Vietnamese `wav2vec2-base` aligners — only `190 MB` each in FP16! — alongside one shared Mimi encoder):
  - Reads each shard's 4 parquets **once**,
  - Runs `gate()` and `audio.decode()` + `soxr(..., 24000)` **once** (cutting decode + resample + I/O CPU by **~45–50%**!),
  - Draws both directions' delays (`vi-en` delay on `en`, `en-vi` delay on `vi`), encodes the 4 clips (`vi_raw`, `en_delayed`, `en_raw`, `vi_delayed`) in one shared sorted Mimi queue (larger pool = tighter batch padding!), runs EN CTC on `en_raw` and VI CTC on `vi_raw`, and writes both `cache/<source>/shard_*.pt` and `cache/en-vi/<source>/shard_*.pt` in parallel!

### D2. Compiled C / Cython / `torchaudio` Forced Alignment (`20–50×` Faster Viterbi)

Target: `training/prep/text_timing.py:163-214`.

- `_ctc_token_frames()` runs a pure Python `for time_index in range(1, time_steps)` loop (`500–1,100` iterations/clip, 6 NumPy ops/iter = `~3–10 ms` CPU per clip).
- Replace the 25-line DP + backtrace inner loop with either:
  - A tiny **C / Cython extension** (`_ctc_viterbi.c`, ~60 lines, compiled once at env build) preserving **100% bit-exact tie-breaking** (stay > advance > skip) and exact score calculation at **`<0.15 ms`/clip (`25–50×` faster)**, OR
  - `torchaudio.functional.forced_align` (C++/CUDA, verify exact tie-breaking & score formula parity).

### D3. Eliminate Per-Row `/dev/shm` File Storms (`37 s` Sys CPU / Shard → `~2 s`)

Target: `training/prep/cache_rows.py:219-252`, `cache_build.py:76-127`.

- Today `prepare_row()` creates and unlinks **1 `.f32` file per row** (`3,218` files/shard → `7.76M` minor page faults and `37.0 s` sys CPU in `build_w14.log`).
- **Fix:** Write **1 contiguous slab file per 32-row decode job** (32× fewer `open/write/read/unlink` syscalls), or use a **fixed preallocated `/dev/shm` ring buffer** (`mmap`'d once per worker and GPU process, zero file creation/deletion, zero page-fault churn!).
- Also keep `pcm` inside the GPU subprocess's direct reader (bypassing the main process entirely — eliminating 1 full `10 MB/row` memory copy in the main process!).

### D4. Multi-Shard Prefetch & Async Shard Writes

Target: `training/prep/cache_build.py:497, 512`, `training/prep/cache_codes.py:168`, `training/tranche/cache.py:140-169`.

- **Multi-shard Hub prefetch:** `Builder._read_ahead` currently reads only `1` shard ahead (`ThreadPoolExecutor(1)`). At `8,000× RT`, a 15 h shard finishes in **~6.7 s**, requiring `~50–70 MB/s` sustained input bandwidth. Add a bounded **2–4 shard download prefetch** (`HF_HUB_DISABLE_XET=1`, `max_workers=8`) onto local disk ahead of `read_shard` (so compressed parquets sit ready on disk without inflating RAM with decoded PCM).
- **Avoid full `.to_pylist()` dict explosion in `read_parts()` (`tranche/cache.py:112`):** Iterate PyArrow columns directly (or filter `vi_gate` / `en_broken` before materializing audio bytes, avoiding allocating Python dicts/bytes for the 15–28% of rejected YODAS/GigaSpeech rows!).
- **Async `write()` in `cache_codes.py`:** Port `tranche/cache.py:247-256`'s background `ThreadPoolExecutor(1)` shard writer into `cache_codes.py:168` (which currently blocks the main loop on `torch.save` + `fsync`).

---

## 6. Part E — Optimizing Noisy Cache Generation (`noisy_cache.py` + `augment.py`)

Measured baseline on RTX 5090: `~2,780–3,182× RT` steady (`2,130×` w/ startup/save on 1 shard, GPU active only **30% of wall time** in `gpu_n14.csv`, mean active power **449 W**).

On top of **Part A (4–5× faster Mimi)**, `noisy_cache.py` and `augment.py` have **five severe pipeline bugs/bottlenecks** unique to them:

| Bottleneck | Location | Root Cause | Fix |
|---|---|---|---|
| **1. Per-clip unbatched GPU augmentation** | `noisy_cache.py:365-372`, `augment.py:240-318` | `torch.from_numpy(pcm).to(device)` does **1,900 blocking pageable H2D copies** per shard; `Bank.apply()` runs **1 clip at a time** (~35,000 tiny FFT/RMS/mask kernel launches/shard!), recomputing `rfft(rir)` and `rfftfreq`/`gain` on every clip | Stage batch into pinned memory (1 async H2D copy); **pre-cache RIR FFTs** per power-of-2 size (or batch-pad RIRs) and **pre-cache the 3 band gain filters** (`16k`, `8k`, `phone`) per `fft_size`; run **batched `[B, fft_size]` `rfft`/`irfft`** + batched noise mix + gain in ~8 GPU kernels per batch instead of ~1,500! |
| **2. 1-row-per-IPC decode jobs & pickle pipe** | `noisy_cache.py:501` | Submits 1 `ProcessPoolExecutor` job per clip (`1,900` IPC round trips/shard) and pickles `~2.3 GB` of float32 PCM through the pool pipe (never got `cache_build.py`'s `DECODE_JOB_ROWS=32` + `/dev/shm` fix!) | Batch **32 clips per decode job** and pass clips via **per-job `/dev/shm` slabs** (same as D3). |
| **3. Synchronous `mimi_batch` D2H drain** | `noisy_cache.py:392`, `cache_build.py:239` | `mimi_batch()` calls `.to("cpu", torch.int32)` synchronously every batch with zero double-buffering or overlap | Reuse `CUDAModels`'s double-buffered pinned H2D/D2H pipelining (`_pipelined` + preallocated pinned output buffers). |
| **4. Synchronous shard `load()` & `torch.save()` on GPU thread** | `noisy_cache.py:525, 537-540` | `pq.read_table`, `torch.load(shard_path)`, and `torch.save(payload)` + `durable()` all block the GPU loop between shards | Move `load()` into a 1-shard read-ahead thread and `torch.save()` + `durable()` + `unlink()` into a background writer thread (just like `tranche/cache.py:247`). |
| **5. `fetch --dataset-dir` 18 GB RAM spike** | `noisy_cache.py:257-274` | `--jobs 16` reads 16 full local parquets into RAM simultaneously (`~18 GB` peak) | Cap local `--dataset-dir` fetch to `--jobs 4` (NVMe-bound, not network-bound) or stream directly from `--dataset-dir` into `encode` without writing/deleting intermediate `.sources/*.parquet` files! |

**Expected `noisy_cache encode` impact:** **`~2,800–3,182× RT → ~10,000–14,000× RT`** (`~3.5–4.5×` faster end-to-end; bounded primarily by PyAV Opus decode + `soxr` on 16–19 CPUs).

---

## 7. Projected Throughput & Cost Summary (Single RTX 5090 @ `$0.511/h`)

| Phase | Mimi Solo (RT) | wav2vec2 CTC Solo (RT) | Grounded Build 1-Dir (RT) | Grounded Build Dual-Dir (`vi-en`+`en-vi`) | Noisy Encode (RT) | 20k h Scale-Up Time & Cost |
|---|---:|---:|---:|---:|---:|---|
| **Current baseline** (`3.14t` no-Triton, TF32, eager `moshi` + `transformers`) | ~3,180× | ~3,300× | **~2,150×** | ~2,150× (2 passes) | ~2,800–3,180× | **~43.3 h (~$22.10)** |
| **Phase 1: Near-lossless Blackwell pass** (GPU subprocess + Triton, FP16+FP32-accum stateless Mimi w/ fused Triton SEANet or TRT + FA2 + fast RVQ, FP16 wav2vec2 w/ Triton masked GroupNorm + varlen FA2 + compile, C Viterbi, `/dev/shm` slabs, GPU `24k→16k`, batched `augment`) | **~13,000–16,000×** | **~14,500–17,500×** | **~6,500–8,500×** | **~7,500–10,000×** (single pass) | **~10,000–13,500×** | **~10.0–12.5 h (~$5.10–$6.40)** |
| **Phase 2: 2-Stream / 2-Worker MPS Overlap + Quality-Gated Blackwell FP8** (`FrozenFP8Linear` on wav2vec2 12L & Mimi 8L transformer, multi-shard Hub prefetch) | **~15,000–18,000×** | **~20,000–24,000×** | **~8,500–10,500×** | **~9,500–12,000×** (CPU decode bound) | **~11,500–15,000×** | **~8.0–10.0 h (~$4.10–$5.10)** |

*(At `~10,000× RT` on a 19.2-CPU RTX 5090 box, building 20,000 h of YODAS both ways + noisy EN→VI drops from **~43 h (`~$22`)** to **~9–10 h (`~$4.60–$5.10`)** — a **~4.3× speedup**.)*

---

## 8. Parity Gates & Recommended Execution Order

Every optimization must pass strict parity checks on the same shards (1 Libriheavy `en-vi` shard `libriheavy-large-00012-00103` + 1 YODAS shard both ways) before merging, following [CONTEXT.md](../CONTEXT.md#essential-invariants):

### Parity Acceptance Criteria

| Component | Metric vs FP32 / Production TF32 Reference | Acceptance Bar |
|---|---|---|
| **Mimi encoder** | Codebook 0 equality; all-16 equality; fp32 decode reconstruction SNR (256 clips) | `cb0 ≥ 99.8%` vs fp32 (matches TF32's `99.92%`); decode SNR within `±0.02 dB` of fp32 (`~9.04 dB`) |
| **wav2vec2 CTC** | Overall & **minimum per-clip** argmax equality (both EN and VI); accept/reject diff; token-frame shifts | **Every clip `≥ 99.0%` argmax**; `< 0.1%` token frames shifted by 1; zero systematic rejection drift |
| **Viterbi (`_ctc_token_frames`)** | Word spans & alignment scores vs NumPy reference | **100% bit-identical** spans, scores within `1e-6` |
| **Batched `augment.Bank.apply`** | Waveform max abs error & Mimi codes vs unbatched `Bank.apply` | Waveform `max_abs_diff < 1e-5`, `cb0 ≥ 99.9%` equal |
| **End-to-end shard** | Row count, IDs, delays, fields, `streams.py` packing & loading | Identical schema/fields, passes `streams.py` pack & load check |

### Incremental Execution Steps (RTX 5090)

1. **Profile baseline kernels (`torch.profiler` + `nvidia-smi` every `0.5 s`)** on 1 window of `libriheavy-large-00012-00103` and 1 YODAS shard to record exact per-op baseline ms on `torch 2.15+cu134` (validating our roofline split between Mimi SEANet/Transformer/RVQ and wav2vec2 Conv/GroupNorm/Transformer).
2. **Build `mimi_cuda.py` (stateless channels-last Mimi + Triton fused SEANet residual + strided GEMM downsamples + FP16/FP32-accum + FA2 + fast RVQ)** and benchmark against a **TensorRT 10.x FP16** engine of SEANet+Transformer; verify `cb0 ≥ 99.8%` and fp32 decode SNR `~9.04 dB`.
3. **Optimize `CUDAModels.ctc_log_probs`** (FP16 + Triton fused masked GroupNorm+GELU + folded `weight_norm` + fused QKV + packed varlen FA2 + `torch.compile`); verify `≥ 99.0%` min per-clip argmax on both EN and VI aligners.
4. **Move GPU runner to dedicated `spawn` subprocess** (enabling Triton/compile/TRT with zero GIL contention), add preallocated double-buffered pinned outputs, C/Cython Viterbi, per-job `/dev/shm` slabs (no delay zero-fill), and verify GPU `24k → 16k` resampling parity for CTC.
5. **Port Mimi + batched GPU `augment.Bank.apply` + 32-row `/dev/shm` decode + async load/save into `noisy_cache.py`**; benchmark on 2 shards interleaved against baseline.
6. **Add `--direction both` dual-direction mode + multi-shard Hub prefetch (`HF_HUB_DISABLE_XET=1`)** to `tranche/cache.py` and benchmark 2-worker MPS vs 2-stream overlap on 4–10 shards.
7. **(Quality-gated Phase 2)** Benchmark `FrozenFP8Linear` on wav2vec2's 12 Transformer layers and Mimi's 8 Transformer layers against the strict parity gates above; keep only if both speedup and parity pass.


---

## 9. Measured RTX 5090 Verification Results (2026-11-28, box `62513802`)

Verified on Vast `62513802` (RTX 5090 32 GB, `sm_120`, torch `2.15.0+cu134`, triton `3.9.0`, 19 CPUs) against stock `main` (`TF32`, `NO_TORCH_COMPILE=1`) on 4 shards × 1,080 real Opus bilingual rows (`4,320` rows, ~25 h Mimi audio per direction, 15 workers):

| Component / Benchmark | Stock `main` (TF32) | Optimized (5090) | Speedup | Parity |
|---|---:|---:|---:|---|
| **Mimi encode (320 s batch)** | `119.23 ms` (`2,684× RT`, `19.79 GiB`) | **`18.63 ms` (`17,175× RT`, `3.15 GiB`)** | **6.40×**, 6.3× less VRAM | `cb0=99.83%`, `all16=93.59%`, decode SNR **`6.795 dB`** (vs stock TF32 `6.745 dB`, Exact FP32 `6.746 dB`) |
| **wav2vec2 CTC `[en]` (320 s)** | `119.76 ms` (`2,672× RT`, `20.56 GiB`) | **`43.00 ms` (`7,442× RT`, `4.01 GiB`)** | **2.79×**, 5.1× less VRAM | argmax vs Exact FP32: `mean=99.889%`, `min=99.387%` |
| **wav2vec2 CTC `[vi]` (320 s)** | `121.81 ms` (`2,627× RT`, `21.51 GiB`) | **`43.41 ms` (`7,371× RT`, `5.07 GiB`)** | **2.81×**, 4.2× less VRAM | argmax vs Exact FP32: `mean=99.945%`, `min=99.552%` |
| **`cache_codes vi-en` (4 shards, 24.61 h)** | `46 s` (`1,939× RT`; shards 1–3 `2,461× RT`) | **`23 s` (`3,791× RT`; shards 1–3 `7,384× RT`)** | **2.00×** total / **3.00×** steady | `4,200/4,200` kept, `30/30` rejected; `99.883%` exact text frames (`max_shift=7fr`); `cb0=99.66%` |
| **`cache_codes en-vi` (4 shards, 25.15 h)** | `44 s` (`2,063× RT`; shards 1–3 `2,614× RT`) | **`25 s` (`3,676× RT`; shards 1–3 `6,797× RT`)** | **1.76×** total / **2.60×** steady | `4,212/4,212` kept, `27/27` rejected; `99.995%` exact text frames (`max_shift=1fr`); `cb0=99.66%` |
| **`DualBuilder` both dirs (8 shards, 49.75 h)** | `95.81 s` (2 passes) | **`41.27 s` (`4,357× RT` total / `~5,400× RT` steady)** | **2.32×** total | `100.000%` kept/rejected match vs single-dir |
| **`noisy_cache encode` (4 shards, 6.11 h)** | `18.22 s` (`2,009–2,990× RT` steady) | **`11.76 s` (`4,965–11,911× RT` steady)** | **1.55×** total / **~3.3×** steady | `1,694/1,694` clean byte-exact, `2,506/2,506` conds match, `cb0=99.74%`, `all16=95.41%` |
| **Finetune `streams.room_codes` (32 × 40 s)** | `2.577 s` (batch 1) | **`0.477 s`** (`FastMimi` + batch 8) | **5.40×** | `cb0=99.93%`, `all16=97.31%` |
| **Finetune `decode_wavs` (8 rows)** | `857.2 ms` | **`244.2 ms`** (batched causal) | **3.51×** | `max_abs_diff = 1.86e-4` |

### Key Empirical Adjustments vs Initial Plan
- **Disabled `torch.compile` on `FastMimiEncoder`'s 8L transformer by default (`compile_transformer=False`):** Eager FP16 runs at `18.63 ms` (`17,175× RT`) vs `17.27 ms` compiled — saving `12–17 s` of compile/recompile overhead on variable shapes (`B = 1` tail batches) for only `1.36 ms`/320 s.
- **Replaced bool attention mask with float16 additive mask + `SDPBackend.CUDNN_ATTENTION` + NHWC `conv2d` for `c1..c6` in `FastWav2Vec2CTC`:** Cut SDPA from `12.30 ms → 4.52 ms` (2.7×) and `c1..c6` from `18.18 ms → 11.45 ms`, bringing total 320 s CTC down from `59.1 ms → 43.0 ms` without requiring `flash_attn`.
- **Kept CPU `soxr` for 16 kHz CTC (`gpu_resample_ctc=False`), FP32 `c0_conv` (`c0_fp16=False`), and FP16 (not FP8) CTC by default:** GPU `Resample24to16` (`min 94.5%` EN / `96.0%` VI) and `--fp8-ctc` (`min 93.9%` EN / `97.6%` VI) missed the `≥99.0%` min per-clip argmax parity bar; FP32 `c0_conv` kept `max_lp_diff` 4× tighter (`0.75` vs `2.99`) at ~4 ms/320 s cost.

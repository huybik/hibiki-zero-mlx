# Student Slices Training Optimization Plan

Date: 2026-11-27

Systems, kernel, Blackwell FP8, and recipe optimization plan for training the 14-layer depformer-slices phone student (`student_slices14`, 9 slices of `512 × 2`, `parallel_codebooks = 8`) on a 32 GB RTX 5090 (`sm_120`).

Related notes: [student-slices](agent-memory/student-slices.md) · [sft-speed](agent-memory/sft-speed.md) · [sft-training](agent-memory/sft-training.md) · [phone-student](agent-memory/phone-student.md) · [voice-scores](agent-memory/voice-scores.md) · [student_architecture_plan](student_architecture_plan.md).

**Status (2026-11-27, PR #205):**
- **Built:** S1–S4, F1, F2 and R1–R3. Not built: S5 (multi-tensor AdamW, batched embeddings).
- **Measured speed** ([student-slices](agent-memory/student-slices.md#training)):
  - Stage 1: 0.428 s an update, against the 0.30–0.38 s projected below.
  - Stage 2: 0.698 s an update, in pieces of 6,144, against the 0.46–0.52 s projected. Whole batches did not fit without checkpointing.
- **How parity was checked:** no unit tests (the repo adds none). Instead, on the box:
  - losses at updates 40–50 against `db0de25`;
  - a resume at update 10;
  - a frozen run's trunk, bit for bit against the cut.
- **The FP8 teacher:** both stages use `--fp8-teacher all`. F2 below planned `trunk` first, gated by a 2k-update A/B against the bf16 teacher. That A/B was not run; 5 batches measured the target shift instead.
- **Stage 1 is running** as `student_slices14_stage1`.

---

## 1. Current Baseline & Diagnosis

Baseline run: `student_slices14_5090` (`restart.sh student-slices-5090`, commit `35e35d9`, RTX 5090 32 GB @ `$0.511/h`), initialized from `both_heads_z_best1200` (14L trunk) + 9 randomly initialized `512 × 2` slices (`134.7M` random params out of `1.346B` total), distilled from the 3B bf16 teacher (`grpo_twoway_best1400`) on `12,288`-frame batches (`--micro-frames 4096`, `--bf16-adamw`, `--lr 5e-5`, 1 epoch = `94,034` updates).

- **Current speed & cost:** `0.99–1.00 s/update` (`~11,800 frames/s`), `26.1 GiB` peak VRAM, `~26.2 h` training + `~1.0 h` validation (12 × `291 s`) ≈ **`27.2 h` (`~$13.90`/epoch)**.

### 1.1 Exact FLOP Budget (12,288 frames/update)

With `dim = 2048`, `kv_repeat = 2`, `hidden_scale = 6` (`gating_hidden = 8192`), `text_card = 48,000`:
- 1 main transformer layer = `62.91M` params (`in_proj 8.39M + out_proj 4.19M + linear_in 33.55M + linear_out 16.78M`).
- `text_linear` = `98.30M` params.
- Teacher depformer (6 layers, `d = 1024`, `hidden = 4096`, 16 steps) = `16 × 6 × 16.78M + 18.9M + 33.6M` = `1,663.1M` params-equivalent per frame.
- Student slices (2 layers, `d = 512`, `hidden = 2048`, 9 slices) = `9 × 2 × 4.19M + 9.4M + 16.8M` = `101.7M` params-equivalent per frame.

| Component | Mode | Params-equiv / frame | TFLOP / update | Share of Matmul FLOPs |
|---|---|---:|---:|---:|
| Teacher 28L main trunk | fwd only | 1,761.6M | 43.3 | **26.0%** |
| Teacher 6L × 16-step depformer | fwd only | 1,663.1M | 40.9 | **24.6%** |
| Teacher `text_linear` | fwd only | 98.3M | 2.4 | **1.5%** |
| *Teacher subtotal* | *fwd only* | *3,523.0M* | *86.6* | ***52.1%*** |
| Student 14L main trunk | fwd + bwd (3×) | 2,642.4M | 64.9 | **39.0%** |
| Student `text_linear` | fwd + bwd (3×) | 294.9M | 7.2 | **4.4%** |
| Student 9 slices (`512 × 2`) | fwd + bwd (3×) | 305.1M | 7.5 | ***4.5%*** |
| *Student subtotal* | *fwd + bwd (3×)* | *3,242.4M* | *79.7* | ***47.9%*** |
| **Total** | | **6,765.4M** | **166.3** | **100.0%** |

> **Key takeaway:** Over **52%** of all matmul work is the frozen 3B teacher forward, **43.4%** is the already-converged student trunk + text head, and only **4.5%** is the newly initialized 9 slices.

### 1.2 Kernel Profile Breakdown (`torch.profiler`, 3 Updates on RTX 5090)

From `/Volumes/data/hibiki-zero-evals/box-62513802/slices-bench-2026-11-27/prof_slices.txt` (`2.921 s` total CUDA = `974 ms/update`, vs heads' `2.820 s`):

| Kernel / Op | CUDA Time (3 updates) | Share | Calls | Root Cause |
|---|---:|---:|---:|---|
| `aten::mm` | 2,184 ms | 74.8% | 9,840 | 50% more GEMM launches than heads (`6,669`) due to 3× micro-batches (`4,096`), 8× repeated teacher slice-8 GEMMs, and unbatched `depformer_in` / `linears` |
| `aten::add_` + `CUDAFunctor_add` | 105.4 ms | 3.6% | 2,850 | Gradient accumulation across 3 micro-batches + unbatched embedding adds |
| `aten::_efficient_attention_forward` | 69.0 ms | 2.4% | 80 | Runs Ampere `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` on Blackwell for `K=9` and `K=16` (86–98% wasted tile work) |
| `aten::_flash_attention_forward` / `_backward` | 89.4 ms | 3.1% | 560 | Main transformer causal GQA attention (already near optimal) |
| `_bf16_sr_adamw_kernel` | 49.1 ms | 1.7% | 916 | 1 Triton launch per parameter tensor (~305/step) |
| `Memset` + `aten::zero_` | 47.0 ms | 1.6% | 5,207 | Zeroing buffers / grad accumulation overhead |
| `aten::embedding_dense_backward` | 27.2 ms | 0.9% | 420 | **459 ms CPU (11.9% of CPU time)**: 42 separate embedding tables (`32` audio + `text` + `9` depformer) |
| `aten::copy_` | 24.6 ms | 0.8% | 4,520 | **442 ms CPU (11.4% of CPU time)**: `_undelay_sequence` full-logit NaN copies & tensor staging |
| `token_kl` reduction kernel | 22.6 ms | 0.8% | 19 | Materializes full fp32 `log_softmax` over `48,000` text vocab and `16 × 2,048` audio vocab |
| `_rope` fused pointwise | 20.8 ms | 0.7% | 420 | Recomputes `arange`, `exp`, `cos`, `sin` inside every transformer layer (42× per micro-batch) |

### 1.3 Why `--micro-frames 6144` OOMed

From `/Volumes/data/hibiki-zero-evals/box-62513802/slices-bench-2026-11-27/bench-slices.log` (lines 112–114):
```text
buf0 = empty_strided_cuda((s82, 1, s93, s0), ..., torch.float32)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.08 GiB.
```
At `micro_frames = 6,144`, `train.py` and `common.py` simultaneously hold:
- `teacher.text_logits` (`[B, 1, T, 48000]` bf16 ≈ `0.59 GiB`) + `student.text_logits` (`0.59 GiB`),
- `teacher.logits` (`[B, 16, T, 2048]` bf16 ≈ `0.40 GiB`) + `student.logits` (`0.40 GiB`),
- `token_kl`'s fp32 `log_softmax` intermediates (`~2.2 GiB` text + `~1.6 GiB` audio).

That **`~5.8 GiB`** of logit + KL buffers forced `student_slices14` down to `--micro-frames 4096` (3 micro-batches/update).

### 1.4 Recipe & Convergence Diagnosis (`student_slices14_5090`, Steps 1–8,100)

500-step bucket averages from `/Volumes/data/hibiki-zero-evals/box-62513802/student_slices14_5090/train_log.jsonl`:

| Steps | LR | Total KL | Semantic (cb 0) | Acoustic (cb 1–15) | Text KL | Content Text KL | PAD Text KL | Grad Norm | s/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10–500 | 5.3e-6 | 3.893 | 3.745 | 3.953 | 0.1215 | 0.1991 | 0.0732 | 3.82 | 1.071 |
| 510–1,000 | 1.1e-5 | 1.672 | 1.377 | 2.717 | 0.1210 | 0.2049 | 0.0690 | 1.43 | 0.998 |
| 1,510–2,000 | 2.1e-5 | 0.852 | 0.552 | 2.122 | **0.0948** | **0.1665** | **0.0503** | 1.45 | 1.001 |
| 3,510–4,000 | 4.3e-5 | 0.625 | 0.320 | 1.834 | 0.1078 | 0.1939 | 0.0546 | 1.56 | 1.000 |
| 5,510–6,000 | 5.0e-5 | 0.598 | 0.283 | 1.697 | 0.1303 | 0.2373 | 0.0639 | 1.66 | 0.997 |
| 7,510–8,000 | 5.0e-5 | 0.559 | 0.252 | 1.573 | **0.1343** | **0.2464** | **0.0650** | 1.54 | 1.002 |
| 8,010–8,100 | 5.0e-5 | 0.557 | 0.252 | 1.547 | **0.1362** | **0.2493** | **0.0656** | 1.53 | 1.007 |

Three clear failure modes:
1. **Trunk drift from high LR (`5e-5`):** The 14L trunk starts already converged (`both_heads_z_best1200`), so text KL starts at `0.114` and bottoms at `0.0948` around `lr ≈ 2e-5`. Pushing `lr` to `5e-5` degrades `content_text_loss` by **+50%** (`0.1665 → 0.2493`), collapsing step-8,000 selecting chrF to **`30.50`** (vs heads' `36.92` / `39.8+`, FLEURS-en `−9.0`) and breaking EOS stopping. (Consistent with `student_dep10_distill`, which stalled at `3e-5` and needed `1e-5` in `student_dep10_anneal`.)
2. **Acoustic starvation (`first_codebook_weight = 100.0`):** Codebook 0 gets `100 / 115 = 87%` of the audio loss weight; each acoustic codebook `1..15` gets `0.87%` (`0.43%` of total loss). Codebook 0 has already converged (`0.252`), while `acoustic_loss` is still `1.55` (`val` TF acoustic CE `4.83 / 3.89`, `mean_hnr_db = 2.9–4.0 dB`, `mean_speaker_sim = 0.10–0.21`), further dampened because `grad_clip = 1.0` clips every step (`grad_norm ≈ 1.55`).
3. **Blind spot between AR `cb 1..7` and parallel `cb 8..15`:** Codebooks `8..15` are predicted in parallel without seeing `8..j-1` of the same frame, giving them an irreducible KL floor vs the teacher. Averaging `cb 1..7` and `cb 8..15` into one `acoustic_loss` hides whether the voice-carrying AR codebooks `1..7` are converging.

---

## 2. Track A — Exact Systems & Kernel Optimizations (Zero Quality Change)

Every optimization in Track A computes the exact same mathematical loss and gradients as the current trainer.

### S1. Chunked Fused Linear + Exact KL Loss (Eliminates ~5.5 GiB VRAM, Unlocks Whole-Batch `12,288`)

**Target files:** `training/finetune/common.py`, `training/student/slices.py`, `training/finetune/lm_training.py`, `training/finetune/train.py`.

**Design:**
- When `teacher is not None` and delays match between student and teacher (exact for `student_slices14`, both `[0, 0, 2, 2, ..., 2]`), add a hidden-state distillation forward path that stops **before** `text_linear` and audio `linears`, returning:
  - Student & teacher `transformer_out` (`[B, T, 2048]`, `25 MB` each at 6,144 frames instead of `590 MB` logits),
  - Student & teacher normed depformer outputs (`[9, B*T, 512]` and `[16, B*T, 1024]`).
- Implement a custom `torch.autograd.Function` (`chunked_linear_kl(h_s, w_s, h_t, w_t, weights, chunk_size=1024)`) using the analytical KL gradient (Liger / Cut Cross-Entropy pattern, **zero recomputation**):
  - For each chunk `c` of `chunk_size` valid (time-aligned) frames:
    1. Compute `z_t = (h_t[c] @ w_t.T).float()` and `z_s = (h_s[c] @ w_s.T).float()` (bf16 matmul, fp32 reduction).
    2. Compute `log_pt = log_softmax(z_t, -1)`, `log_ps = log_softmax(z_s, -1)`, `pt = exp(log_pt)`, `ps = exp(log_ps)`.
    3. Accumulate exact weighted loss `kl_sum += (w[c] * (pt * (log_pt - log_ps)).sum(-1)).sum()`.
    4. Exact logit gradient is a single subtraction: `dz_s = (w[c].unsqueeze(-1) * (ps - pt)).to(h_s.dtype)` — **no `log_softmax_backward` needed**.
    5. Immediately compute `dh_s[c] = dz_s @ w_s` and accumulate `dw_s += dz_s.T @ h_s[c]`.
    6. Drop `z_t`, `z_s`, `pt`, `ps`, `dz_s` before the next chunk.
  - `backward(grad_out)` simply returns `(dh_s * grad_out, dw_s * grad_out, None, ...)`.
- Apply to both `text_linear` (`vocab = 48,000`) and the 16 audio `linears` (fusing `linears[8..15]` into one `[16384, 512]` projection per chunk).
- Because delays match (`0` for text/cb0, `2` for cb1–15), apply the `T - delay` slice directly on the hidden states before chunking — eliminating `_undelay_sequence`'s full-logit `aten::copy_` and `NaN` fills.

**Expected impact:**
- **VRAM:** `−5.0 to −5.5 GiB` peak (peak at `6,144` drops from `>31.4 GiB` OOM to `~23 GiB`, and **whole `12,288`-frame batches** fit with `--grad-checkpoint-layers 0..2` or `--fp8-teacher`).
- **Speed:** `~10–14%` faster (fewer micro-batches eliminates `aten::add_` gradient accumulation and `aten::zero_`, triples GEMM `M` dimension, and eliminates full-vocab HBM round-trips).

**Verification:** Unit test on tiny models comparing `chunked_linear_kl` loss, `dh_s`, and `dw_s` against `token_kl` in fp32 (`max_abs_diff < 1e-5`) and bf16 (within normal bf16 rounding).

---

### S2. Batch Teacher Depformer Steps `8..15` in `_per_step`

**Target file:** `training/finetune/lm_training.py` (`_per_step`, lines 123–125).

**Problem:** The 3B teacher's `depformer_weights_per_step_schedule` is `[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]`. Currently `_per_step` runs `modules[8]` **8 separate times** per projection (`in_projs`, `out_projs`, `gating.linear_in`, `gating.linear_out`) across all 6 layers = **192 separate GEMMs** reading slice-8 weights 8× from HBM.

**Design:**
- Group contiguous runs of identical indices in `schedule` inside `_per_step`:
  - For a run `start..end` sharing `index`, flatten `xs[start:end]` to `[(end - start) * N, C]`, execute `function(modules[index], flat)` once, and reshape back to `[end - start, N, out]`.
- Eliminates **168 GEMMs per micro-batch** (`504/update` at 3 micro-batches), cuts slice-8 weight HBM reads by **8×**, and raises `M` from `4,096` to `32,768` (`98,304` at whole batch `12,288`).

**Expected impact:** `~5–8%` faster per step, ~15 lines of code, bit-identical math.

---

### S3. Batched `depformer_in` & `linears[8..15]` + Compiled `slices._forward_training`

**Target files:** `training/student/slices.py` (`_forward_training`, lines 70–82), `training/finetune/lm_training.py` (`forward_depformer_training`, lines 188–215).

**Design:**
1. Fuse the 9 `depformer_in[k](transformer_out)` calls into **1 GEMM** (`F.linear(transformer_out, torch.cat([m.weight for m in self.depformer_in], 0))`) for both student (`2048 → 4608`) and teacher (`2048 → 9216`), reading `transformer_out` once instead of 9×.
2. Fuse student's `linears[8..15](normed[8])` into **1 GEMM** (`512 → 16384`), or feed directly into S1's chunked KL.
3. Wrap the pre-depformer and post-depformer norm/embedding/add logic in `@torch.compile(dynamic=True)` so `depformer_norms` (9 LayerNorms) and embeddings do not dispatch eagerly in Python.
4. Benchmark `torch.bmm` / `torch._grouped_mm` for the student's 9 narrow `d = 512` slices inside `_depformer_layer` vs unbind on RTX 5090 (keep only if faster on `sm_120`).

**Expected impact:** `~3–5%` faster per step.

---

### S4. Blackwell Depformer Attention (`K=9`, `K=16`) & Cached RoPE Tables

**Target file:** `training/finetune/lm_training.py` (`_depformer_layer` line 141, `_rope` lines 278–291).

**Design:**
1. **Depformer attention:** Currently pinned to `SDPBackend.EFFICIENT_ATTENTION`, which dispatches an Ampere `64×64` tile kernel (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`, `69 ms` / 3 updates) for `K = 9` (student) and `K = 16` (teacher). Benchmark on RTX 5090:
   - Compiled explicit causal math attention (`q @ k.T * scale`, upper-tri mask, softmax, `@ v`) — for `K = 9` (`81` scores/head) and `K = 16` (`256` scores/head), Inductor/Triton avoids 64×64 tile waste;
   - `SDPBackend.FLASH_ATTENTION` and `SDPBackend.CUDNN_ATTENTION` on `sm_120`;
   - Keep the fastest kernel with exact bf16 parity.
2. **Cached RoPE `(cos, sin)`:** Precompute `(rotr, roti)` once up to `context = 3,000` per `(device, dim, max_period)` and slice `[:steps]` instead of recomputing `arange`/`exp`/`cos`/`sin` 42× per micro-batch (`20.8 ms` / 3 updates).

**Expected impact:** `~2–3%` faster per step.

---

### S5. Multi-Tensor Triton `BF16StochasticAdamW` & Batched Audio Embeddings

**Target files:** `training/finetune/bf16_adamw.py`, `training/finetune/lm_training.py`.

**Design:**
1. **Multi-tensor AdamW launch (or CUDA-graphed step):** Currently launches ~305 individual Triton kernels per step (`49 ms` / 3 updates + Python loop overhead). Grouping same-dtype parameters into a persistent/multi-tensor Triton kernel (or pointer-array launch) eliminates per-tensor launch overhead (matters more once step time drops below `0.6 s`).
2. **Batched `emb[0..31]` lookup in `forward_text`:** Moshi loops 32 times over `self.emb[k](codes[:, k])`, causing 32 separate embedding forwards, 32 `add_` ops, and 32 `embedding_dense_backward` radix sorts (`459 ms` CPU / `27 ms` CUDA per 3 updates). Benchmark a single compiled offset-indexed lookup over stacked `emb` views (keeping `state_dict` keys unchanged).

**Expected impact:** `~2–3%` faster per step and significantly lower CPU overhead.

---

## 3. Track B — Blackwell (`sm_120`) FP8 Compute

On RTX 5090 (`sm_120`), rowwise E4M3 `torch._scaled_mm` achieves **`650–735 TFLOP/s`** vs **`380–444 TFLOP/s`** in bf16 (`~1.65×`).

### F1. Fused Triton Rowwise FP8 Linear for Student Main Trunk (`--fp8`)

**Target files:** `training/finetune/fp8_linear.py`, `training/finetune/lm_training.py`.

**Problem with current `fp8_linear.py`:**
- `_quantize_rowwise` runs 3 eager PyTorch ops (`x.float()`, `abs().amax()`, `/scale.clamp.to(fp8)`) writing and re-reading full fp32 buffers to GDDR7.
- Backward performs 3 full HBM transpose copies (`w_2d.T.contiguous()`, `dy_2d.T.contiguous()`, `x_2d.T.contiguous()`).
- `_FP8RowwiseLinear.apply` causes Dynamo graph breaks inside `_compiled_transformer_layer`, preventing cross-op fusion with `_rms_norm` and `silu(gate) * value`.

**Design:**
1. Write two fast single-pass Triton kernels in `fp8_linear.py`:
   - `triton_quantize_rowwise_fp8(x_bf16) -> (x_fp8, scale_fp32)`: reads bf16 once into SRAM/registers, computes row amax, writes E4M3 + fp32 scale directly (no fp32 HBM intermediate).
   - `triton_transpose_quantize_rowwise_fp8(x_bf16) -> (xt_fp8, scale_fp32)`: fused tiled transpose + rowwise FP8 quantize for `dX` and `dW` in backward (eliminates `.T.contiguous()` HBM passes).
2. Register the kernels via `torch.library.custom_op` so `torch.compile` traces cleanly across layers without graph breaks.
3. Optionally fuse `_rms_norm` → `quantize_rowwise_fp8` and `silu(gate)*val` → `quantize_rowwise_fp8` for zero extra HBM round-trips.

**Trade-off & impact:**
- **Zero shift in teacher targets** (only student trunk matmuls use FP8; 50-step parity already within `+0.002` loss).
- **Speed:** `~10–14%` faster per step on `student_slices14` (speeds up the `39%` student trunk matmuls).

---

### F2. Pre-Quantized FP8 Frozen Teacher (`--fp8-teacher trunk|all|none`)

**Target files:** `training/finetune/fp8_linear.py`, `training/finetune/lm_training.py`, `training/finetune/common.py`, `training/finetune/train.py`.

**Design:**
- The frozen 3B teacher accounts for **52% of all FLOPs** (`86.6 TFLOP/update`), and its weights **never change**.
- Quantize teacher linear weights **once at startup** into `(wq_e4m3 [N, K], sw_fp32 [N, 1])` and drop the bf16 weight copies:
  - `--fp8-teacher trunk` (**recommended default to test first**): quantizes only the teacher's 28 main layers (`112` linears). Keeps the teacher's 6L depformer and `text_linear` in **exact bf16**, preserving clean acoustic/voice targets (`KL(bf16 ‖ fp8)` previously measured at only `0.0037` text / `0.010` audio).
  - `--fp8-teacher all`: quantizes all `354` teacher linears (trunk + depformer; `KL(bf16 ‖ fp8)` = `0.005` text / `0.023` audio).
- At runtime, teacher forward does **zero weight quantization** and only 1 fused Triton activation quantization per linear (and for slice-8 batched steps `8..15`, one large `[32768, 1024]` FP8 GEMM).

**Trade-off & impact:**
- **VRAM:** `−1.76 GiB` (`trunk`) or `−3.13 GiB` (`all`), giving comfortable headroom for whole `12,288`-frame batches.
- **Speed:** `~10–13%` (`trunk`) or `~18–22%` (`all`) faster per step.
- **Gate:** Validate on a 2k-step A/B probe against bf16 teacher on teacher-forced loss, chrF, `mean_speaker_sim`, and `mean_hnr_db` before enabling in production.

---

## 4. Track C — Training Recipe & Two-Stage Schedule (Biggest Total-Cost + Quality Multiplier)

Fixing the step-8,000 trunk drift and acoustic starvation both **improves quality** and **cuts total updates by 2–3×**.

### R1. Stage 1: Frozen-Trunk Slice Warmup (`--freeze-trunk`)

**Target files:** `training/finetune/train.py`, `training/finetune/common.py`.

**Design:**
- Add `--freeze-trunk`: freezes every parameter outside `slices.AUDIO_PARTS` (`transformer.*`, `text_emb`, `text_linear`, `out_norm`, `emb.*`), leaving only the `134.7M` slice parameters trainable.
- When `--freeze-trunk` is active:
  1. Run the student's 14L main trunk under `torch.no_grad()` — **zero trunk backward pass** (`−26%` of total FLOPs) and **zero saved trunk activations** (`−8 to −10 GiB` VRAM).
  2. Skip `text_linear` and `text_kl` for **both** teacher and student (`−6%` of FLOPs, `−3.5 GiB` VRAM).
  3. Allocate AdamW moments only for the `134.7M` trainable slice params (`−4.8 GiB` optimizer/grad VRAM).
  4. Train at a higher slice LR (`2e-4` to `3e-4`, short warmup) for `~8,000–12,000` updates.
- Note: Even with the trunk frozen under the new `[0, 2, 2, ..., 2]` target delays, step-10 `text_loss` was already `0.114` — confirming `transformer_out` is immediately clean and usable for warming up the random slices without corrupting the trunk.

**Expected impact:**
- **Stage 1 step time:** `~0.42–0.50 s/step` (bf16 + S1–S4) or **`~0.30–0.38 s/step`** (with F2 `--fp8-teacher`) — **2.5–3.3× faster** than current `1.00 s/step`.
- **Stage 1 wall time:** `10,000` updates in **`~0.9–1.3 hours`** (`~$0.50–0.65`), delivering warmed-up slices while keeping 100% of `both_heads_z_best1200`'s text translation quality.

---

### R2. Stage 2: Joint Fine-Tuning with Differential LR (`--trunk-lr`) & WSD Schedule

**Target files:** `training/finetune/train.py`, `training/finetune/bf16_adamw.py`.

**Design:**
- Add `--trunk-lr` (default: same as `--lr`), splitting optimizer parameters into two `param_groups`:
  - `trunk` group (non-`AUDIO_PARTS`): peak `1e-5` (or `5e-6–1e-5`, matching `student_dep10_anneal` and staying safely below the `~2.5e-5` drift threshold seen in `student_slices14_5090`).
  - `slices` group (`AUDIO_PARTS`): peak `5e-5` (or `1e-4`).
- `train.py`'s LR loop scales each group by its own `base_lr * schedule_multiplier`.
- Use `--schedule constant` with validation every `8,000` updates (checking both text chrF and voice scores), then branch a short `--schedule decay` (`1 - sqrt` cooldown, e.g. `8k–16k` updates) once validation plateaus — avoiding paying for a full `94,034`-update epoch when `30k–50k` updates suffice.

**Expected impact:**
- Eliminates the `+50%` text KL regression and chrF collapse (`39.8 → 30.5`).
- Cuts Stage 2 length from `94,034` updates to `~30,000–50,000` updates (**~2–3× fewer updates**).

---

### R3. Acoustic Loss Rebalancing & Split `cb 1..7` / `cb 8..15` Diagnostics

**Target files:** `training/finetune/common.py`, `training/finetune/train.py`.

**Design:**
1. **Split logging (always on, zero cost):** Log `acoustic_ar_loss` (`cb 1..7`, autoregressive) and `acoustic_par_loss` (`cb 8..15`, parallel) separately in `train_log.jsonl` and `val_log.jsonl`. Because `cb 8..15` have an irreducible KL floor relative to the teacher, tracking `cb 1..7` directly tells us whether the voice-carrying codebooks are converging.
2. **Acoustic weighting lever:** Add `--acoustic-ar-weight` (default `1.0`, weight on `cb 1..7` vs `1.0` on `cb 8..15`) and tune `--first-codebook-weight`:
   - **Stage 1 (`--freeze-trunk`):** `--first-codebook-weight 10.0 --acoustic-ar-weight 3.0` (gives `cb 0` = 10, `cb 1..7` = 21 total, `cb 8..15` = 8 total → `cb 1..7` receive **54%** of audio loss instead of **6.1%**!).
   - **Stage 2 (joint):** `--first-codebook-weight 20.0 --acoustic-ar-weight 2.0` (or test `100` vs `20` on an 8k probe so `cb 0` / semantic timing stays sharp while `cb 1..7` keep strong gradient signal).
3. **Grad clipping check:** Log clipped-step fraction and separate trunk vs slices gradient norms (already available on GPU before AdamW step, zero sync per step) to ensure trunk/cb0 gradients aren't clipping away slice `1..8` updates.

---

## 5. Projected Speed, Memory, and Cost Summary (RTX 5090 @ `$0.511/h`)

| Configuration | Micro-Frames | Peak VRAM | Est. s/update | Est. frames/s | 10k Stage 1 + 40k Stage 2 | Full 94k Epoch | Cost (2-Stage) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Current baseline** (`35e35d9`, 1-stage 94k) | 4,096 (3×) | 26.1 GiB | 1.00 s | 11,800 | — | 27.2 h | **$13.90** (1-stage) |
| **+ Track A (S1–S5, exact bf16)** | 12,288 (1×) | ~24–26 GiB | ~0.72–0.78 s | ~15,500 | ~10.3 h | ~20.5 h | **~$5.30** |
| **+ F1 (Student fused FP8)** | 12,288 (1×) | ~24–26 GiB | ~0.62–0.68 s | ~18,000 | ~8.8 h | ~17.5 h | **~$4.50** |
| **+ F2 (`--fp8-teacher trunk`)** | 12,288 (1×) | ~22–24 GiB | ~0.52–0.58 s | ~21,000 | **~7.1 h** | ~14.7 h | **~$3.65** |
| **+ F2 (`--fp8-teacher all`)** | 12,288 (1×) | ~21–22 GiB | ~0.46–0.52 s | ~23,500 | **~6.2 h** | ~13.0 h | **~$3.15** |
| *Stage 1 frozen trunk alone (S1–S5 + F2 trunk)* | *12,288 (1×)* | *~12–14 GiB* | *~0.32–0.38 s* | *~33,000* | *(1.0 h for 10k)* | — | *(~$0.50)* |

*(Estimates include ~30 min for 6 validations in the 2-stage run and ~1 h for 12 validations in a full 94k epoch.)*

---

## 6. Execution & Verification Protocol

Following [CONTEXT.md](../CONTEXT.md#essential-invariants) (profile and verify parity before any full run; never modify `/workspace/hibiki-zero` while a run is active):

1. **Develop in a dedicated worktree/branch** (e.g. `slices-opt` off `35e35d9`) with CPU unit parity tests for:
   - `chunked_linear_kl` vs `token_kl` (exact fp32 forward & gradients, exact match with `--micro-frames` denominators),
   - `_per_step` contiguous run batching (exact match vs unbatched on teacher schedule `[0..7, 8×8]`),
   - fused `depformer_in` / `linears[8..15]` and cached RoPE (exact match vs current `slices.py`),
   - `--freeze-trunk`, `--trunk-lr`, `--acoustic-ar-weight`, and `cb 1..7` / `cb 8..15` logging,
   - resume round-trip (`RESUME_KEYS` updated cleanly for new flags).
2. **Incremental RTX 5090 benchmark & `torch.profiler` pass** (60–100 updates on the same rows as `slices-bench-2026-11-27`, `nvidia-smi` every `0.5 s`, peak VRAM + `s/step` + 50-step loss parity):
   - Baseline (`4,096` micro) → `+ S1–S4` (`6,144` and `12,288` whole) → `+ S5` → `+ F1 (student FP8)` → `+ F2 (--fp8-teacher trunk` and `all)` → `Stage 1 (--freeze-trunk)`.
   - Keep only levers that show measured speedup without breaking parity.
3. **Short quality & voice probe (~1.5–2 hours on RTX 5090):**
   - Run Stage 1 (`--freeze-trunk`, `8,000` updates, `lr = 2e-4`, `--first-codebook-weight 10 --acoustic-ar-weight 3`) and validate text + voice (`mean_speaker_sim`, `mean_hnr_db`, `cb 1..7` vs `cb 8..15` KL).
   - Run `8,000` updates of Stage 2 (`--trunk-lr 1e-5 --lr 5e-5`) and compare directly against `student_slices14_5090`'s step-8,000 validation (`30.50` selecting chrF, `0.15 / 2.9 dB` FLEURS-vi).
   - If F2 (`--fp8-teacher trunk`) is selected, verify its 2k-step KL & voice scores match the bf16 teacher.
4. **Full optimized run & post-training:**
   - Launch Stage 1 → Stage 2 (`constant`, val every `8,000`), branch `decay` once selecting chrF & voice plateau, then run on-policy `both_z` + 6-bit palettization (`palettize.py` extended to slices) + Core ML ANE export.

# VI→EN Text Translation Optimization Plan (RTX 5090 / Blackwell)

Date: 2026-11-28

Systems, Blackwell (`sm_120`) NVFP4/FP8 kernel, memory-hierarchy, and algorithmic optimization plan for speeding up VI→EN dataset translation (`training/tranche/translate.py`, `IndexTeam/Index-Translate-35B-A3B-preview-FP4` on vLLM) on a single 32 GB RTX 5090 (`sm_120`, 575 W TDP, 1.79 TB/s GDDR7, 209.5 BF16 / 419 FP8 / 838 NVFP4 dense TFLOPS).

Related notes: [tranche-translate](agent-memory/tranche-translate.md) · [tranche-yodas](agent-memory/tranche-yodas.md) · [tranche-box](agent-memory/tranche-box.md) · [vieneu_asr_5090_optimization_plan](vieneu_asr_5090_optimization_plan.md).

---

## 1. Current Baseline & Roofline Diagnosis

### 1.1 Measured RTX 5090 Baselines

Measured on box `62513802` (RTX 5090, 32 GB, driver `595.91.07`) running `training/tranche/translate.py` with `IndexTeam/Index-Translate-35B-A3B-preview-FP4`@`111ff5e9` in vLLM 0.34 (`gpu_memory_utilization=0.95`, `max_model_len=4096`, `max_num_seqs=256`, `max_num_batched_tokens=16384`, `enable_prefix_caching=False`, `limit_mm_per_prompt={"image": 0, "video": 0}`):

| Dataset | Avg VI chars / row | Est. prompt / output tokens | Measured throughput | GPU telemetry | 20,000 h YODAS scale-up (~5.2M rows) |
|---|---:|---:|---:|---|---:|
| **GigaSpeech 2** (archive 60, `21,810` rows) | ~70 chars | ~115 in / ~20 out | **97 rows/s** (`225 s`) | 92% busy; chrF 90.3 vs FP8 | *(shorter clips)* |
| **YODAS pilot** (23 shards, `264,665` rows, `1,012 h`) | **213 chars** (~3× Giga) | **~268 in / ~60 out** | **44.2 rows/s** (`99.8 min`) | ~89% busy, **~460 W** (vs 575 W TDP), ~137k KV tokens | **~32.8 5090-hours (~$16.80 @ $0.51/h)** |

**Strategic impact:** In [vieneu_asr_5090_optimization_plan.md](vieneu_asr_5090_optimization_plan.md), optimizing VieNeu-TTS + Qwen3-ASR cuts 20,000 h YODAS voicing from ~41–50 h down to **~12–15 5090-hours**. Without optimizing translation, VI→EN text translation at **~33 5090-hours** becomes the **largest GPU-hour stage in the entire data-generation pipeline** (>2× TTS + ASR combined).

### 1.2 Exact Architecture & VRAM Anatomy (`Index-Translate-35B-A3B-preview-FP4`)

From the checkpoint's `config.json`, `hf_quant_config.json`, and file manifest (`111ff5e950f75dcccb1e631f65e936781a9c34d7`, `Qwen3_5MoeForConditionalGeneration`, quantized by NVIDIA ModelOpt 0.44 NVFP4 W4A4, `group_size=16`):

- **Hybrid backbone (40 layers, `hidden=2048`):**
  - **30 `linear_attention` (Gated DeltaNet / GDN) layers:** `16` K heads × `128`, `32` V heads × `128`, `conv_kernel=4`, `mamba_ssm_dtype: "float32"`.
  - **10 `full_attention` layers** (`full_attention_interval=4`): `16` Q heads / `2` KV heads, `head_dim=256`, `attn_output_gate=true`.
  - **40 MoE layers:** `256` routed experts + `1` shared expert (`moe_intermediate_size=512`, `top-8` routed + `1` shared = `9` active experts/token → **~2.41B** active linear params/tok + `0.51B` `lm_head` on decode).
  - **Native MTP weights:** `model_mtp.safetensors` (`1.69 GB`, `mtp_num_hidden_layers=1`) is shipped in the repo (unreferenced in `model.safetensors.index.json`, unquantized BF16).

| Component | Precision in Checkpoint | Param Count | VRAM Weight Footprint | Per-Token Active FLOPs |
|---|---|---:|---:|---:|
| **40 MoE layers** (`256+1` experts) | **NVFP4** (4.5 b/param w/ FP8 scales) | 32.34B (1.13B active) | **16.94 GiB** (`18.19 GB`) | 2.26 GFLOPs (NVFP4) |
| **10 `full_attention` layers** | **NVFP4** | 0.27B | **0.14 GiB** | 0.55 GFLOPs (NVFP4) |
| **30 `linear_attention` layers** (`ignore` list!) | **Unquantized BF16** | **1.01B** | **1.88 GiB** (`2.01 GB`) | **2.01 GFLOPs (BF16!)** |
| **Embeddings + `lm_head`** (untied, `248,320 × 2048`) | **Unquantized BF16** | 1.02B | **1.89 GiB** (`2.03 GB`) | **1.02 GFLOPs (BF16, decode)** |
| **Vision tower `model.visual.*`** (27 layers, `ignore` list!) | **Unquantized BF16** | ~0.42B | **~0.78 GiB** | 0 (unused) |
| **Total weights** (`model-00001..3.safetensors`) | Mixed NVFP4 + BF16 | ~35.1B | **21.73 GiB** (`20.85 GiB` LM-only) | **4.82 GFLOPs prefill / 5.84 decode** |

**Recurrent state & KV cache per sequence:**
- **30 GDN `linear_attention` layers:** SSM state `(32, 128, 128) = 524,288` elements/layer + conv state `(8192, 3)` in BF16 (`1.41 MiB` total):
  - At `bfloat16` SSM state: `30.0 + 1.41 =` **`31.4 MiB / seq`** (`232 seqs × 31.4 MiB = 7.11 GiB`, closely matching `tranche-translate.md`'s observation that at `0.90` utilization only 232 sequences fit).
  - At `float32` SSM state (`config.json` specifies `mamba_ssm_dtype: "float32"`): `60.0 + 1.41 =` **`61.4 MiB / seq`** (if enforced, doubles both VRAM footprint and per-step state traffic).
- **10 `full_attention` layers:** `10 × 2 × 2 × 256 = 10,240` elements/token = **`20 KiB/tok` in BF16** (`10 KiB/tok` in FP8).
- In vLLM's hybrid KV manager, the `~137k` token KV budget observed on the 5090 corresponds directly to `256` active slots × `~535` aligned tokens/slot alongside the 30 linear-attention state groups.

### 1.3 Root-Cause Bottlenecks (Why YODAS Runs at Only 44 rows/s)

| # | Bottleneck | Magnitude / Evidence | Roofline Impact |
|---|---|---|---|
| **1** | **76% prompt bloat & 3× redundant prefill** | Each YODAS row sends ~73 boilerplate tokens (`PROMPT` + chat template + `<think>\n\n</think>`) + **~130 tokens of `CONTEXT_ROWS=2`** for only ~65 tokens of target segment (`4.5:1` input:output ratio). Every segment is prefilled **3 times** across consecutive prompts. | **~82% of all processed tokens are prefill** (~11,850 prefill tok/s vs ~2,650 output tok/s). |
| **2** | **256-expert MoE bandwidth wall at `B = 256`** | At `B = 256` and `top-8` of 256 experts, $1 - (1 - 8/256)^{256} = \mathbf{99.97\%}$ of experts fire every decode step, reading all **18.19 GB of MoE weights** for only **8 tokens/expert** (arithmetic intensity ~32 FLOPs/B vs RTX 5090 NVFP4 ridge point ~467 FLOPs/B). | Decode is deeply **memory-bandwidth bound** (~12 ms/step just to stream MoE weights, plus ~10–20 ms/step to read+write GDN recurrent states). |
| **3** | **Unquantized BF16 `linear_attn` + `lm_head` + unused vision tower** | All 30 dense `linear_attn` layers (`1.01B` params, `1.88 GiB`) sit in the checkpoint's `ignore` list and run in **BF16 (209.5 TFLOPS)** instead of FP8 (419) or NVFP4 (838) — **42% of all prefill FLOPs** (and 52% of decode FLOPs w/ `lm_head`). `model.visual.*` wastes ~0.78 GiB if loaded without `language_model_only=True`. | Cuts Tensor Core speed by 2–4× on nearly half the active FLOPs and steals ~2–3 GiB VRAM that would otherwise unlock `max_num_seqs=512`. |
| **4** | **`sm_120` NVFP4 kernel fallback risk** | Consumer Blackwell (`sm_120`, 99 KiB shared mem vs `sm_100` B200's 228 KiB) frequently falls back to **Marlin W4A16** (dequantizing FP4→BF16 at 209.5 TFLOPS) or suboptimal CUTLASS tiles in vLLM 0.33/0.34 unless explicitly verified. `envs.sh` still pins `vllm==0.33.0`. | Can silently cost **1.3–1.8×** throughput and explain the ~460 W draw vs 575 W TDP. |
| **5** | **Unused native MTP head (`model_mtp.safetensors`)** | The checkpoint already includes a 1-layer Multi-Token Prediction head, unused by `translate.py`. Even at `B = 256`, verifying `2–3` tokens/step keeps MoE experts well below the NVFP4 compute ridge (`8 → 24` tokens/expert) while amortizing the 18.2 GB MoE weight read. | Leaves **1.3–1.6×** lossless greedy decode speedup on the table. |
| **6** | **`translate.py` tail drain, static `MAX_TOKENS=512`, unsorted batches** | `MAX_TOKENS=512` lets runaway rows decode 400+ wasted steps even though `bad()` rejects anything > `2.5 × len(vi) + 20` chars (~110 tokens for 200-char VI). `GROUP=4` drains the batch `256 → 1` every 4 shards and runs small retry passes sequentially. Unsorted prompts mix 60-char and 500-char rows. | Wastes 10–20% wall time on tail drains, runaway stragglers, and CPU-side chat formatting/tokenization gaps. |

---

## 2. Part A — Zero-Quality-Change Engine, VRAM & Blackwell (`sm_120`) Optimizations

All items in Part A keep the exact same model predictions (greedy output unchanged, or strictly validated to ≤0.1 chrF drift for cache dtypes).

### A1. Reclaim VRAM & Unlock `max_num_seqs = 512` (`translate.py:125-142`)

Doubling decode concurrency from `256 → 512` doubles tokens per MoE expert (`8 → 16`) with **zero increase in the 18.19 GB MoE weight read per step**. Today 32 GB VRAM blocks `512` sequences. Reclaim VRAM via four levers:

1. **Pass `language_model_only=True` to `LLM(...)`** (replaces/augments `limit_mm_per_prompt={"image": 0, "video": 0}`):
   - `limit_mm_per_prompt` skips multimodal requests/processors, whereas `language_model_only=True` prevents vLLM from loading `model.visual.*` (~0.78 GiB BF16 weights) into VRAM and skips multimodal profiling.
2. **Enable FP8 full-attention KV cache (`kv_cache_dtype="fp8"` or `"fp8_e4m3"`):**
   - Halves the 10 full-attention layers' KV footprint (`20 → 10 KiB/tok`) and halves full-attention KV bandwidth on FlashInfer `sm_120` kernels, freeing ~1.3 GiB (or doubling the 137k-token KV pool).
3. **Verify & tune `mamba_ssm_cache_dtype`:**
   - First inspect vLLM's startup log on the 5090 to see whether `mamba_ssm_cache_dtype="auto"` currently resolves to `float32` (`61.4 MiB/seq`) or `bfloat16` (`31.4 MiB/seq`).
   - Because translation sequences are very short (~150–600 tokens, no long-context reasoning), explicitly setting `mamba_ssm_cache_dtype="bfloat16"` (or `"float16"` if kernels prefer it) cuts recurrent state VRAM and per-step state read/write bandwidth (`30.7 → 15.4 GB/step` at `B=256`) by **2×** if vLLM was respecting `config.json`'s `float32`. Validate chrF parity on FLEURS dev + Archive 50.
4. **Tighten `max_model_len` (`4096 → 1024` for single-row, or `2048` for multi-row batching):**
   - A single-row YODAS prompt (~268 avg, ~650 max) + dynamic `max_tokens` (A4.1) never approaches 4,096 tokens. Lowering `max_model_len` shrinks activation/CUDA-graph scratchpad reservations and lowers the hybrid cache minimum floor.
5. **Raise `max_num_seqs` from `256 → 384 / 512`** once the freed ~2–4 GiB fits the additional GDN state slots. Keep `enable_prefix_caching=False` for single-row prompts:
   - *Why prefix caching never hit before:* `PROMPT` + chat header is only ~73 tokens, whereas vLLM's aligned hybrid Mamba/GDN block size is ~512–1,024 tokens — a 73-token prefix cannot fill even one aligned block, so it paid a ~3× memory penalty for a 0% hit rate.

### A2. Audit & Tune `sm_120` NVFP4 MoE Kernels, vLLM Config & Alternative Runtimes

1. **Upgrade `envs.sh` `mt` env past `vllm==0.33.0` (`>=0.34+` / latest CUDA 13.x Blackwell release)** with matching `flashinfer-jit-cache` (`VLLM_HAS_FLASHINFER_CUBIN=1`) and `TORCH_CUDA_ARCH_LIST=12.0`.
2. **Verify active NVFP4 MoE & GEMM backends in startup logs** (and benchmark all `sm_120` options on 2,000 YODAS rows):
   - Check whether vLLM is running native FP4 Tensor Core grouped GEMMs (`flashinfer_cutlass` / `cutlass`) or silently falling back to `marlin` (W4A16, dequantizing to 209.5 TFLOPS BF16).
   - Benchmark explicitly across:
     - `--moe-backend flashinfer_cutlass` (or `flashinfer`) vs `cutlass` vs `marlin` (`VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4`).
     - `max_num_batched_tokens` sweep: `4096`, `8192`, `16384`, `32768` (larger prefill chunks push MoE + `linear_attn` further into the compute-bound regime, critical when input:output is 4.5:1!).
     - `async_scheduling=True` (vLLM V1 CPU-GPU scheduler overlap — eliminates the CPU-bound engine loop overhead previously observed in `tranche-translate.md:23`) and full/piecewise CUDA graph capture up to `max_num_seqs`.
3. **Head-to-head runtime bake-off on RTX 5090 (same checkpoint, same 2,000 YODAS rows):**
   - **SGLang** (Blackwell `sm_120` NVFP4 + FlashInfer/CUTLASS + zero-overhead overlap scheduler + native Qwen3.5 GDN/MTP support): routinely beats vLLM by **10–30%** on high-concurrency offline MoE batches on RTX 5090.
   - **TensorRT-LLM** (`trtllm-bench` / Python `LLM` API): static fused NVFP4 MoE + GDN engine for `sm_120`; worth testing if vLLM/SGLang remain below ~520 W or leave NVFP4 Tensor Cores underutilized.

### A3. Native MTP Speculative Decoding (`model_mtp.safetensors`) — Lossless Greedy Speedup

- `IndexTeam/Index-Translate-35B-A3B-preview-FP4` **already ships `model_mtp.safetensors` (`1.69 GB`)** (`mtp_num_hidden_layers: 1`).
- **Why MTP can STILL win at `B = 256` on a 256-expert MoE:**
  - Unlike dense models that become compute-bound by `B = 64`, a 256-expert top-8 MoE at `B = 256` only feeds **8 tokens/expert** (`AI ≈ 32 FLOPs/B` vs `467 FLOPs/B` NVFP4 ridge).
  - With `speculative_config={"method": "mtp", "num_speculative_tokens": k}` (`k = 1, 2, 3`), verifying `k+1` tokens in parallel feeds `16–32` tokens/expert — still memory-bandwidth-bound — while amortizing the **18.19 GB MoE weight read** across `~1.7–2.5` accepted tokens/step!
  - Greedy VI→EN translation has very low entropy and high MTP acceptance (~75–88%).
  - **Mathematical guarantee:** Greedy (`temperature=0`) MTP verification produces **100% bit-identical output tokens** to standard greedy decoding.
- **Practical caveats to verify on the 5090:**
  1. `model_mtp.safetensors` is currently separate from `model.safetensors.index.json` and stored in **BF16 (`1.69 GB` = 1 unquantized 256-expert layer)**. Check whether vLLM loads it directly or needs a local merged index, and note the `1.57 GiB` VRAM cost: enable MTP **after** A1 reclaims VRAM (and ideally after quantizing the MTP MoE layer to NVFP4/FP8 → `~0.46 GB`, see B2!).
  2. Benchmark `k = 1, 2, 3` at `B = 128, 192, 256` — if `linear_attn` BF16 compute or GDN verification overhead saturates before MoE bandwidth gains pay off, MTP will show its exact crossover point empirically.

### A4. Fix `translate.py` Pipeline, Dynamic `max_tokens` & Tail-Drain Bottlenecks

Targeted software fixes in `training/tranche/translate.py` (all zero quality change):

1. **Dynamic per-row `max_tokens` tied to `bad()`'s length gate (`translate.py:41, 107, 155`):**
   - Today every request uses `MAX_TOKENS = 512`, even though `bad()` rejects any with-context reply longer than `2.5 * len(vi) + 20` characters (~110 English tokens for a 200-char Vietnamese row).
   - A looping/runaway row currently burns **~400 wasted decode steps**, stalling the entire shard tail drain, only to be discarded by `bad()`!
   - **Fix:** Set per-request `max_tokens` dynamically (e.g., `min(512, max(64, int((2.5 * len(vi) + 40) / 3.0)))` with context, and `min(512, max(96, int((4.0 * len(vi) + 60) / 3.0)))` without context — conservative 3.0 chars/token floor so no valid translation is ever clipped). Aborts looping rows **3–5× sooner**.
2. **Length-sorted submission & continuous cross-shard streaming (`translate.py:51, 176, 247-284`):**
   - Today `GROUP = 4` submits unsorted rows and drains the GPU from `256 → 1` at the end of every 4-shard group, then executes up to 5 tiny retry batches (`attempt = 1..5`, often 5–50 rows at <10% GPU utilization) synchronously before starting the next group.
   - **Fix:**
     - Sort `todo` by Vietnamese character length (descending) before submission so long rows start early and short rows backfill the tail (or group similar lengths together to minimize GDN/prefill padding).
     - Increase `GROUP` from `4 → 12–16` shards (or switch from blocking `llm.chat()` to vLLM's `LLMEngine` / `AsyncLLMEngine` streaming loop that keeps `max_num_seqs` pinned at 100% across shard boundaries and writes each shard's `.parquet` as soon as its rows finish).
     - **Defer retries (`attempt >= 1`)** into a single batched pass per super-group (or end of run) rather than running 5 tiny GPU calls after every 4 shards.
3. **Background CPU chat-template formatting & tokenization (`TokensPrompt`):**
   - Pre-apply `chat_template.jinja` and `tokenizer.encode()` on a background CPU thread pool while the GPU translates the previous group, passing pre-tokenized `TokensPrompt(prompt_token_ids=...)` to `llm.generate()` instead of calling `llm.chat()` synchronously on the main thread (~4 s/group saved + eliminates Python `SamplingParams` construction overhead by sharing a small dict of bucketed `max_tokens` `SamplingParams` objects!).
4. **Skip rows already excluded by upstream gates & deduplicate identical `(context, text)` prompts:**
   - YODAS ingest already applies `vi_gate`, so all `yodas/segments/` rows are kept; for sources with un-gated segments where `asr_vi` already exists (e.g., `giga`), skip rows that fail `vi_gate` before translating. Deduplicate exact `(prompt)` matches within a super-group (zero quality change).

---

## 3. Part B — Blackwell Custom Quantization Pass (Eliminating Unquantized BF16 Hotspots)

Look back at Section 1.2: `IndexTeam/Index-Translate-35B-A3B-preview-FP4` leaves **~4.5 GiB** of weights in **unquantized BF16**, including **all 30 dense `linear_attention` layers** — which represent **42% of prefill FLOPs** and run on the RTX 5090's slowest Tensor Core tier (**209.5 TFLOPS BF16** vs **419 TFLOPS FP8** and **838 TFLOPS NVFP4**).

### B1. Quantize the 30 `linear_attention` Linear Projections to FP8 W8A8 (Hybrid NVFP4-MoE + FP8-GDN Checkpoint)

- **What to quantize:** Only the dense linear projections inside `layers.{i}.linear_attn` (`in_proj_qkvz` / `in_proj_qkv` / `in_proj_z`, `out_proj` = `1.01B` params, `1.88 GiB` BF16) → **FP8 E4M3 W8A8** (using NVIDIA ModelOpt or `llm-compressor` calibrated on ~512 Vietnamese→English YODAS + FLEURS prompts, starting from the existing NVFP4 checkpoint or the BF16 base).
- **What stays in BF16/FP32:** `conv1d`, `in_proj_ba` (decay/beta gating), `A_log`, `dt_bias`, norms, MoE routers, `shared_expert_gate`, and the GDN recurrence itself.
- **Why FP8 is safe here:** The official FP8 build (`Index-Translate-35B-A3B-preview` online FP8) already quantizes `linear_attn` linear layers to FP8 and scored **65.1 / 37.7** on FLEURS dev (higher than the NVFP4 checkpoint's 64.5 / 36.4!).
- **Expected impact:**
  - **VRAM saved:** **~0.94 GiB** (directly unlocks ~30 more GDN sequence slots on 32 GB).
  - **Compute & bandwidth:** Doubles Tensor Core throughput on 42% of prefill FLOPs (`209.5 → 419 TFLOPS`) and halves `linear_attn` weight reads (`2.01 → 1.01 GB/step`).
  - **Net speedup:** **~1.15–1.25×** end-to-end (plus indirect concurrency gain from the freed GiB).

### B2. Quantize `model_mtp.safetensors` to NVFP4 & Test FP8 `lm_head`

1. **`model_mtp.safetensors` (BF16 `1.69 GB` → NVFP4 `~0.46 GB`):**
   - Its 256-expert MoE layer (`1.61 GB` BF16) makes MTP expensive in both VRAM and draft-step bandwidth. Quantizing the MTP MoE layer to NVFP4 (matching the main 40 layers) saves **~1.2 GB VRAM** and makes the draft step ~3× faster — making A3 (native MTP at `B=256`) dramatically more attractive on a 32 GB card, with **zero impact on output quality** (MTP only affects acceptance rate; target verification guarantees identical greedy tokens).
2. **`lm_head` (`248,320 × 2048` BF16 = `1.02 GB` → FP8 `0.51 GB`, quality-gated):**
   - On decode, `lm_head` is read on every single step (`1.02 GB` + `1.02 GFLOPs/tok` in BF16). Quantizing `lm_head` to FP8 frees `0.51 GiB` VRAM and halves vocab projection time. Must check FLEURS dev chrF/BLEU and non-Latin/diacritic failure rate, as `lm_head` is sensitive to rare tokens.

---

## 4. Part C — Algorithmic & Prompt-Level Multipliers (Biggest Single Win: 2–3×)

Even after perfect Blackwell kernel tuning, **single-row translation with `CONTEXT_ROWS=2` forces the GPU to process ~268 input tokens for every ~60 output tokens** on YODAS — re-reading every segment 3 times and repeating a 73-token instruction wrapper 5.2 million times.

### C1. Small-Window Multi-Segment Batching (`K = 4..8` Consecutive Rows per Prompt)

- **Why revisit multi-line packing now:** The 2026-11-06 bake-off (`tranche-translate.md:116-123`) only tested **40-line** batches on **Index-Translate-9B** (and Hy-MT2-30B-A3B), where 40 lines caused word drift across lines (`63.8` vs `64.8` FLEURS chrF). It was **never tested on `Index-Translate-35B-A3B`** (which has `instTrans IFscore 0.8336`, RL-trained specifically for format/structure preservation) nor at **small windows (`K = 4..8`)**.
- **Token math on YODAS (6 consecutive rows of the same `doc`, 1 carry-over context row):**

| Setup (per 6 YODAS rows) | Boilerplate + template | Context tokens | Segment input tokens | Total Prefill Tokens | Output Tokens | Total Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **Current (`K = 1`, `CONTEXT_ROWS = 2`)** | `6 × 73 = 438` | `6 × 130 = 780` | `6 × 65 = 390` | **1,608** | 360 | **1,968** |
| **Packed (`K = 6`, `1` carry-over context row)** | `73` | `65` | `390 + ~24` (`n\|`) | **552 (-66%!)** | `360 + ~18` | **930 (-53%!)** |
| **Packed (`K = 8`, `1` carry-over context row)** | `73` | `65` | `520 + ~32` | **690 (-68%/row)** | `480 + ~24` | **1,214 (-54%/row)** |

- **Compound hardware benefits on RTX 5090:**
  1. **2.9× fewer prefill tokens** (eliminates the 3× duplicate segment prefill and 83% of boilerplate).
  2. **6× longer output per sequence** (`~378` tokens vs `~60`): amortizes GDN state setup and keeps all `256–512` decode slots steadily populated with far fewer prefill/decode phase switches.
  3. **Better cross-segment consistency:** The model sees its *own* previous lines' translations inside the same window (natural terminology/pronoun consistency across consecutive segments of a YouTube video), rather than only untranslated Vietnamese context.
- **Safety net (preserves all `bad()` invariants):**
  - Format: `1|<vi_1>\n2|<vi_2>\n...\nK|<vi_K>` (same strict `parse()` pattern proven in `training/prep/translate_en_speech.py:186-196`).
  - If a packed reply has wrong line count, misnumbered lines, or *any* line fails `bad()` (non-Latin script, Vietnamese echo, bracket, length ratio), **accept the valid lines or fall back to single-row `translate.py` for just that window** (typically <2–5% of windows).
- **Mandatory quality gate before adopting:** Sweep `K ∈ {1, 2, 4, 6, 8, 12}` on:
  1. **FLEURS dev** (149 sentences, grouped sequentially) — target: within `±0.2` chrF of `K=1` (`64.5`).
  2. **Archive 50 (2,000 rows)** + **1,000 consecutive YODAS rows** — measure chrF against `K=1` 35B-A3B FP4, parse failure rate, cross-line word-bleed rate (spot-check 50 pairs + length-ratio outliers), and downstream VieNeu TTS `broken%`.

### C2. Context & Prompt Trimming (Applies Even If `K = 1` Is Kept)

If multi-segment packing causes unacceptable cross-line drift on spoken fragments, two smaller levers still cut single-row prefill by **35–50%**:

1. **Cap YODAS context to `CONTEXT_ROWS = 1` (or last ~100–120 chars of the previous row):**
   - GigaSpeech 2 rows average ~70 chars, so `CONTEXT_ROWS = 2` was only ~140 chars (~40 tokens).
   - YODAS rows average **213 chars** (~13 s of speech), so `CONTEXT_ROWS = 2` injects **426 chars (~130 tokens)** — 2× the target segment! Dropping to 1 prior row (or its final sentence / last 120 chars) cuts YODAS prompt length from `~268 → ~170` tokens (**-37% prefill**). Test chrF on FLEURS dev + Archive 50 + YODAS sample.
2. **Test a shorter/native Chinese system instruction:**
   - `PROMPT` is currently ~55 English tokens. Index-Translate's native training instruction is ~18 Chinese tokens; a terse bilingual/English equivalent preserving the 4 hard rules (fragment fidelity, normal case/punctuation, ASCII Vietnamese names, no brackets) fits in ~30–35 tokens (~20 tokens saved × 5.2M rows = 104M fewer prefill tokens). Validate diacritic-retention rate (currently 0.73%) and bracket rate on 2,000 YODAS rows.

---

## 5. Part D — Quality-Gated Model Alternatives & Roofline Comparison on RTX 5090

Per the project's standing cost rule (`tranche-box.md:157-160`: *"minimize rental cost, not wall time ... quality may change for speed if it stays good"*), compare the full Index-Translate family on a **32 GB RTX 5090**:

| Model (Quantization) | Active GFLOPs/tok (excl. `lm_head`) | Weight VRAM (LM-only) | GDN State / Seq (`bf16`) | Max Safe `B` on 32 GB | Bottleneck Regime on 5090 | Bake-off Quality (FLEURS dev / Arch 50 chrF) |
|---|---:|---:|---:|---:|---|---|
| **Index-Translate-35B-A3B** (current NVFP4, BF16 `linear_attn`) | **4.82** (42% BF16) | 20.85 GiB | 31.4 MiB | 256 → **512** (w/ A1) | MoE weight BW + GDN state BW + BF16 `linear_attn` | **64.5** (FP4) / **70.5** (FP8) |
| **Index-Translate-35B-A3B** (NVFP4 + FP8 `linear_attn`, Part B) | **4.82** (0% BF16) | **19.9 GiB** | 31.4 MiB | **512** | Balanced FP4/FP8 compute & BW | ~64.5–65.0 *(verify)* |
| **Index-Translate-9B** (FP8 W8A8) | **13.82** (2.87× more!) | ~11.1 GiB | 24.0 MiB | 512–768 | **Compute-bound** at 419 TFLOPS FP8 (~30k tok/s ceiling) | **64.8 / 68.7** (BF16/FP8) |
| **Index-Translate-9B** (custom NVFP4 via ModelOpt) | **13.82** | **~8.0 GiB** | 24.0 MiB | **768–1024** | **FP4 Tensor Core** (838 TFLOPS, ~60k tok/s ceiling; 2.6× less weight read than 35B!) | *(unmeasured — must test NVFP4 degradation)* |
| **Index-Translate-2B** (FP8 / BF16) | ~2.4 | ~1.8–3.6 GiB | ~9 MiB | 1024+ | Very fast (>300 rows/s), **too weak standalone** | `FLORES_minor` XCOMET 0.48 vs 0.72 (rejected) |

**Key roofline insights:**
1. **Why 9B FP8 lost to 35B-A3B FP8 on H100/H200, yet 9B NVFP4 is worth testing on RTX 5090:**
   - On 80–141 GB Hopper GPUs, 35B-A3B FP8 ran at `B = 512` with only `4.82 GFLOPs/tok`, easily beating 9B's `13.82 GFLOPs/tok` (`110` vs `84 rows/s`, `tranche-translate.md:128`).
   - On a **32 GB RTX 5090**, 9B quantized to **NVFP4** (`~8.0 GiB` weights vs `21.7 GiB`) runs on 838 TFLOPS FP4 Tensor Cores, reads **2.6× fewer weight bytes per decode step**, and easily sustains `B = 512–768`. Combined with multi-segment packing (`K = 6`), **9B NVFP4** could approach **~180–250 YODAS rows/s**, and in the bake-off 9B BF16 actually tied/beat 35B-A3B FP4 on FLEURS dev (`64.8` vs `64.5`, though Archive 50 was `68.7` vs `70.5`). **Caution:** dense 9B models can lose more quality under W4A4 NVFP4 than 35B MoE models — calibrate with ModelOpt on VI→EN data and check FLEURS dev + Archive 50 + off-target/Chinese-leak rate before considering it.
2. **Distilling a task-specific 2B–4B VI→EN translator (only if scaling to 50k+ h):**
   - If the dataset ever scales well beyond 20,000 h, fine-tuning a small dense model (e.g., `Index-Translate-2B` or `Qwen3.5-4B`) on 500k–1M of our own 35B-A3B VI→EN pairs (specifically matching our exact spoken-fragment style, ASCII names, no brackets) would run at **250–400+ rows/s** on a 5090. Not needed for 20,000 h if Phases 1–3 below already cut translation to ~7–10 hours (~$4–$5).
3. **Co-residency / MPS note:**
   - Unlike VieNeu-TTS (~11 GiB) + Qwen3-ASR (~6–8 GiB), `Index-Translate-35B-A3B-FP4` needs `~29–30 GiB` of the 5090's 32 GB to sustain high MoE batch sizes. Keep running translation **solo on the GPU** (turning off voice/ASR during bursts, as `translate.py` already does); only 9B NVFP4 (`~8 GiB`) could ever share a 5090, and even then solo burst runs avoid MPS contention and simplify failure recovery.

---

## 6. Combined Throughput & Cost Projections (Single RTX 5090)

Projections for **YODAS** (avg 213 VI chars, ~60 EN output tokens/row, ~260 rows/hour of audio; `20,000 h ≈ 5.2M rows`):

| Phase | Changes Included | Est. YODAS Speed | Est. GigaSpeech 2 Speed | 20,000 h YODAS GPU-Hours | Est. Cost (@ $0.51/h) | Quality Impact |
|---|---|---:|---:|---:|---:|---|
| **Baseline (today)** | `35B-A3B-FP4`, vLLM 0.34, `B=256`, `K=1`, `CONTEXT_ROWS=2`, `MAX_TOKENS=512` | **44 rows/s** | **97 rows/s** | **32.8 h** | **$16.75** | Reference (`64.5` FLEURS chrF) |
| **Phase 1: Zero-quality-change systems pass (A1–A4)** | `language_model_only=True`, `kv_cache_dtype="fp8"`, verified `mamba_ssm_cache_dtype="bfloat16"`, `max_model_len=1024`, `max_num_seqs=512`, verified `sm_120` FP4 MoE backend (vLLM ≥0.34 or SGLang), `async_scheduling`, dynamic `max_tokens`, sorted streaming `GROUP=12..16` + deferred retries + pre-tokenized prompts | **~70–95 rows/s** (`1.6–2.1×`) | **~150–200 rows/s** | **15.2–20.6 h** | **$7.75–$10.50** | **None** (identical greedy output) |
| **Phase 2: Blackwell custom quant + native MTP (B1–B2 + A3)** | FP8 W8A8 on the 30 `linear_attn` layers (`+0.94 GiB` VRAM, 2× Tensor Core rate on 42% of FLOPs) + NVFP4 `model_mtp.safetensors` (`k=1..2` lossless greedy MTP if net positive at `B=256..512`) | **~90–130 rows/s** (`2.0–3.0×`) | **~190–260 rows/s** | **11.1–16.0 h** | **$5.65–$8.15** | **Minimal / None** (verify FP8 `linear_attn` on FLEURS/Arch 50; MTP is lossless) |
| **Phase 3: Multi-segment window packing (C1, `K=6..8`)** | Pack `6–8` consecutive `doc` rows/prompt + 1 carry-over context row (`-66%` prefill tokens) with automatic `K=1` fallback on any failed window, stacked on Phases 1–2 | **~160–240 rows/s** (`3.6–5.5×`) | **~300–450 rows/s** | **6.0–9.0 h** | **$3.05–$4.60** | **Quality-gated** (must stay within `±0.2` chrF on FLEURS & Archive 50, zero line bleed) |
| **Phase 4 (Optional fallback/alternative): `Index-Translate-9B` NVFP4 + `K=6`** | ModelOpt NVFP4 9B (`~8 GiB` weights, `B=768`, 838 TFLOPS) + `K=6` packing | **~180–260 rows/s** (`4.1–5.9×`) | **~350–500 rows/s** | **5.5–8.0 h** | **$2.80–$4.10** | **Quality-gated** (must verify NVFP4 9B vs 35B-A3B FP4 on FLEURS + Archive 50 + YODAS) |

---

## 7. Recommended Benchmarking & Execution Order

Every change should be benchmarked on the **same fixed benchmark slice** (e.g., **YODAS shard 0000**, ~11,500 rows, or its first 2,000 rows for rapid sweeps) alongside the **quality gate** (FLEURS dev 149 sentences + Archive 50 2,000 rows + non-Latin / Vietnamese-echo / fail-all-tries rate):

1. **Step 0 — Profile current baseline on RTX 5090 (10 min):**
   - Capture vLLM startup logs (`VLLM_LOGGING_LEVEL=DEBUG` or INFO) on `Index-Translate-35B-A3B-preview-FP4` to confirm:
     1. Which NVFP4 MoE backend was selected (`flashinfer_cutlass`, `cutlass`, or `marlin`).
     2. Whether `model.visual` weights were loaded (VRAM before/after `language_model_only=True`).
     3. What `mamba_ssm_cache_dtype` resolved to (`float32` vs `bfloat16`) and exact KV/Mamba cache size & max concurrency.
     4. Steady-state `nvidia-smi` power/SM%/mem% + vLLM per-step prefill vs decode timing (`--disable-log-stats=False`).
2. **Step 1 — Zero-code-risk vLLM config sweep (A1 + A2):**
   - Test `language_model_only=True`, `kv_cache_dtype="fp8"`, `mamba_ssm_cache_dtype="bfloat16"`, `max_model_len=1024`, `max_num_seqs=512`, `max_num_batched_tokens ∈ {8192, 16384, 32768}`, `async_scheduling=True`, and all `moe_backend` options (`flashinfer_cutlass`, `cutlass`, `marlin`) in `vllm>=0.34` with `flashinfer-jit-cache`.
   - Run a 15-minute head-to-head check of **SGLang** on the same 2,000 YODAS rows — adopt whichever runtime wins clearly on rows/s & stability.
3. **Step 2 — `translate.py` pipeline fixes (A4):**
   - Implement dynamic per-row `max_tokens`, descending length sort, background pre-tokenization (`TokensPrompt`), shared `SamplingParams`, larger `GROUP=12..16` (or streaming engine loop), and deferred retry pass. Verify bit-identical outputs on greedy non-runaway rows.
4. **Step 3 — Multi-segment window packing A/B test (C1, plus C2 fallback):**
   - Prototype `K ∈ {1, 2, 4, 6, 8}` consecutive `doc` rows with 1 carry-over context row and automatic `K=1` fallback on any bad/misaligned window.
   - Score on FLEURS dev, Archive 50, and 1,000 YODAS rows (chrF, BLEU, fallback %, spot-check 50 pairs for line drift, VieNeu TTS broken%). This is the highest-leverage single test (`~2×` standalone).
5. **Step 4 — Custom Blackwell quantization & native MTP (B1 + B2 + A3):**
   - Use NVIDIA ModelOpt (`0.44+`) to produce a local checkpoint adding **FP8 W8A8 on the 30 `linear_attn` linear layers** (and NVFP4 on `model_mtp.safetensors`).
   - Benchmark solo (no MTP) and with native MTP `num_speculative_tokens ∈ {1, 2, 3}` at `B ∈ {128, 256, 384, 512}`. Keep MTP only if net wall-time rows/s improves at peak batch size.
6. **Step 5 (Optional) — `Index-Translate-9B` NVFP4 check (Part D):**
   - Only if Steps 1–4 don't already clear **~150+ YODAS rows/s** (~10 h / ~$5 for 20k h): quantize `Index-Translate-9B` to NVFP4 with ModelOpt and compare speed + FLEURS/Archive 50/YODAS chrF against optimized 35B-A3B.

---

## 8. Measured RTX 5090 Results (2026-11-28, Box `62513802`)

Benchmarked solo on box `62513802` (RTX 5090, 32 GB, driver `595.91.07`) with `vllm==0.34.0`, `torch==2.14.0+cu130`, `flashinfer==0.7.3+cu130`, and `IndexTeam/Index-Translate-35B-A3B-preview-FP4`@`111ff5e9`. Fixed slices: **FLEURS dev** (149 sentences), **YODAS shard 0000** (first 4,000 rows, avg 212.7 VI chars / 51.6 output tokens, scored against the hub `text_en/0000.parquet` reference), and **GigaSpeech 2 Archive 50** (first 2,000 lowercased rows, avg 80.1 VI chars / 16.9 output tokens, scored lowercase against Muse's English). Raw logs archived at `/Volumes/data/hibiki-zero-evals/box-62513802/translate-5090opt-2026-11-28/`.

### 8.1 Tokenizer Prompt-Length Measurements

| Prompt mode | YODAS 0000 avg in tok/row (max) | YODAS reduction | Giga 50 avg in tok/row | Giga 50 reduction |
|---|---:|---:|---:|---:|
| `K=1, context_rows=2` (baseline) | 241.0 (407) | — | 141.9 | — |
| `K=1, context_rows=1, max_context_chars=120` | 166.7 | −30.8% | — | — |
| `K=4, window_context_rows=1` | 99.2 | −58.8% | — | — |
| `K=6, window_context_rows=1` | 86.6 | −64.1% | 45.3 | −68.1% |
| `K=8, window_context_rows=1` | 82.7 (644) | −65.7% | 39.7 | −72.0% |

### 8.2 Full 14-Arm RTX 5090 Sweep

| Arm | Configuration | YODAS (4k) rows/s | Speedup | Arch 50 (2k) rows/s | Speedup | FLEURS chrF / BLEU | Arch 50 chrF / BLEU | YODAS vs ref chrF | Fail (4k) | GPU util / Mean W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm0_baseline` | `main` (`max_model_len=4096`, `seqs=256`, `auto` KV/SSM, `K=1 ctx=2`) | **42.5** | 1.00× | **97.5** | 1.00× | 64.75 / 37.33 | 70.07 / 41.60 | 87.65 | 0 | 96.2% / 436 W |
| `arm1_p1a_safe` | `language_model_only`, `max_model_len=2048`, `seqs=256`, `async_scheduling`, `row_max_tokens`, LPT, dedup, `TokensPrompt` | **43.3** | 1.02× | **102.0** | 1.05× | 64.67 / 37.09 | 70.09 / 41.94 | 87.68 | 0 | 96.6% / 439 W |
| `arm2_fp8kv_512/384` | `kv_cache_dtype=fp8`, `mamba_ssm=auto` (`float32`), `seqs=512/384` | FAILED | — | — | — | — | — | — | — | `Mamba cache blocks (317 / 339)` |
| `arm2_fp8kv_256` | `kv_cache_dtype=fp8`, `mamba_ssm=auto`, `seqs=256` | **39.7** | 0.93× | **94.3** | 0.97× | 64.40 / 36.48 | 70.32 / 42.20 | 87.47 | 0 | 95.7% / 423 W |
| **`arm3_bf16ssm_512`** *(new 5090 default)* | `arm1` + **`mamba_ssm_cache_dtype=bfloat16`** + **`kv_cache_dtype=fp8`** + **`max_num_seqs=512`** | **50.5** | **1.19×** | **110.0** | **1.13×** | 64.47 / 36.74 | **70.44 / 42.20** | 87.58 | 1 | 91.2% / 433 W |
| **`arm6_moe_fi_cutlass`** | Same as `arm3`, explicit `--moe-backend flashinfer_cutlass` | **51.8** | **1.22×** | — | — | 64.67 / 36.94 | — | 87.52 | 0 | 91.0% / 436 W |
| `arm8_moe_marlin` | Same as `arm3`, `--moe-backend marlin` | **47.8** | 1.12× | — | — | 64.81 / 37.01 | — | 88.22 | 1 | 94.9% / 492 W |
| `arm5_moe_b12x` | Same as `arm3`, `--moe-backend b12x` | **40.0** | 0.94× | — | — | 64.52 / 36.46 | — | 87.54 | 0 | 66.7% / 329 W |
| `arm5_moe_fi_b12x` | Same as `arm3`, `--moe-backend flashinfer_b12x` at `seqs=512` | OOM (−38 MiB) | — | — | — | — | — | — | — | `Tried 128 MiB, 89.7 MiB free` |
| `arm7_moe_fi_cutedsl` | `--moe-backend flashinfer_cutedsl` | UNSUPPORTED | — | — | — | — | — | — | — | `does not support current device` |
| `arm4_batched32k` | Same as `arm3`, `--max-num-batched-tokens 32768` | FAILED | — | — | — | — | — | — | — | `Mamba cache blocks (505)` |
| **`arm12_ctx1`** | `arm3` + `--context-rows 1 --max-context-chars 120` | **59.6** | **1.40×** | **121.9** | **1.25×** | **64.80 / 36.91** | 69.96 / 41.53 | 82.57 | 0 | 96.1% / 448 W |
| **`arm9_win4`** | `arm3` + `--window 4 --window-context-rows 1` (99.7% / 99.8% accepted) | **70.7** | **1.66×** | **168.6** | **1.73×** | 64.39 / 36.34 | **70.87 / 45.42** | 77.81 | 0 | 96.9% / 446 W |
| **`arm10_win6`** | `arm3` + `--window 6 --window-context-rows 1` (99.4% / 99.8% accepted) | **70.3** | **1.65×** | **179.1** | **1.84×** | 64.62 / 36.51 | **70.76 / 45.22** | 76.41 | 0 | 93.3% / 432 W |
| **`arm11_win8`** | `arm3` + `--window 8 --window-context-rows 1` (98.9% / 100.0% accepted) | **70.6** | **1.66×** | **189.0** | **1.94×** | **64.78 / 37.06** | **71.38 / 46.52** | 75.75 | 0 | 94.7% / 430 W |
| `arm13_mtp1` | `arm3` + `--mtp-tokens 1` | FAILED | — | — | — | — | — | — | — | `NVFP4 weight_scale ... never loaded` |

### 8.3 Key Empirical Discoveries

1. **Hybrid Mamba/KV page alignment was the true `max_num_seqs=512` blocker:** By default, vLLM 0.34 honors `config.json`'s `mamba_ssm_dtype: "float32"` (`61.4 MiB/seq`). Passing `kv_cache_dtype="fp8"` alone forced the hybrid attention block size to **2,112 tokens**, capped Mamba cache blocks at **317–339**, and slowed `seqs=256` down (`43.3 → 39.7 rows/s`). Pairing **`mamba_ssm_cache_dtype="bfloat16"`** with **`kv_cache_dtype="fp8"`** halves both page sizes in lockstep (keeping attention block size at `1,088` tokens with `1.49%` padding), unlocks **`max_num_seqs=512`** (`117,145` KV tokens, `57.2×` concurrency at `max_model_len=2048`), and boosts identical single-row translation by **+19–22% on YODAS (`42.5 → 50.5–51.8 rows/s`)** and **+13% on Archive 50 (`97.5 → 110.0 rows/s`)** with no quality loss.
2. **Vision tower already skipped in vLLM 0.34:** Both `limit_mm_per_prompt={"image": 0, "video": 0}` and `language_model_only=True` loaded **20.95 GiB** (vs 21.73 GiB full VLM); `translate.py` prefers `language_model_only=True` when supported.
3. **Blackwell `sm_120` NVFP4 MoE backend:** Auto-selection picks **`FLASHINFER_CUTLASS`**, which beat `MARLIN` (`47.8 rows/s`, 492 W) and `B12X` (`40.0 rows/s`). `FLASHINFER_CUTEDSL` is unsupported on `sm_120`; `FLASHINFER_B12X` OOM'd by 38 MiB at `seqs=512` (worth retesting at `seqs=384` in a future pass).
4. **Native MTP requires checkpoint repacking:** `model_mtp.safetensors` is BF16 and not listed in `quantization_config.ignore` or `model.safetensors.index.json`, so vLLM raises `NVFP4 weight_scale for layer 'column_parallel_linear' was never loaded` until the MTP layer is either added to `ignore` or quantized to NVFP4 (Section 3, B2).
5. **Multi-segment window packing (`--window 4..8`) scales cleanly on 35B-A3B:** Unlike the 2026-11-06 40-line 9B bake-off, `35B-A3B-FP4` achieved **98.9–99.7% acceptance on YODAS** and **99.8–100.0% on Archive 50**, reaching **70.7 rows/s (1.66×) on YODAS** (on a 4k slice where 633–1,020 windows have partial tail-drain; full 10k+ shard groups should run faster) and **168.6–189.0 rows/s (1.73–1.94×) on Archive 50** while improving Archive 50 scores (**70.76–71.38 chrF / 45.22–46.52 BLEU** vs baseline `70.07 / 41.60`). Default remains `--window 1` until a full-shard YODAS side-by-side audit confirms zero cross-line phrasing drift; `--window 4..8` and `--context-rows 1 --max-context-chars 120` (`59.6 rows/s`, `1.40×`) are available as CLI flags.

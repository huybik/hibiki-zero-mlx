# Next Phone Student Architecture

Date: 2026-11-27

Design analysis and architecture plan for the next phone student, replacing the 16-codebook parallel-heads student (`student_heads14_cb16`) after voice evaluation showed severe voice-preservation and harmonic degradation despite strong text translation and Apple Neural Engine (ANE) speed.

**Status (2026-11-27):** the student in training is the slices student ([student-slices](agent-memory/student-slices.md)): the heads student's 14-layer trunk with 9 depformer slices of `512 × 2` (codebooks 0–7 one at a time, 8–15 together from the last slice) at the teacher's delays. That is Candidate 2's step layout at Candidate 1's width, trained from random slices rather than cut from the 3B. On the A15 it held 67 ms a frame for 10 minutes ([lm-npu](agent-memory/lm-npu.md#depformer-slices)). Candidate 1's grouping is a later lever if the frame needs room.

Related notes: [student-heads](agent-memory/student-heads.md) · [phone-student](agent-memory/phone-student.md) · [lm-npu](agent-memory/lm-npu.md) · [student-palettize](agent-memory/student-palettize.md) · [on-policy-distillation](agent-memory/on-policy-distillation.md) · [vi_en_parity_plan](vi_en_parity_plan.md).

---

## 1. Current Evidence

| Model | Trunk / Audio Head | Codebooks | Test chrF (11 sets) | Speaker Sim (WeSpeaker) | HNR (dB) | A15 Phone Frame (10 min) |
|---|---|---:|---:|---:|---:|---|
| 3B Teacher (`grpo_twoway_best1400`) | 28L / 6L depformer (`d=1024`) | 16 | 45.34 | 0.36 (cb 0–7: 0.30) | 14.4 (cb 0–7: 14.5) | 135–203 ms (GPU, fails RT) |
| Depformer student (`g2000_q6`) | 10L / 3L depformer (`d=1024`) | 8 | 39.48 | **0.32** | **12.2** | 66–142 ms (MLX GPU, throttles) |
| Heads student (`pal6c4000` / `both_heads_z` 1200) | 14L / 16 linear heads | 16 | **42.53** | **0.00–0.15** (cb 0–7: 0.10–0.11) | **4.2–6.6** (cb 0–7: 6.7–8.1) | **52–54 ms flat** (Core ML ANE) |

Reference input audio HNR on the 5 Vietnamese mic clips is `11.0 dB` ([student-heads](agent-memory/student-heads.md#codebooks)).

Key findings from the voice investigation:
- **Not a quantization, phone, or GRPO artifact:** plain `s16k_q6`, `s112k_pal6_bf16`, `pal6c4000_bf16`, and the Core ML 6-bit release on the phone all exhibit the same voice collapse (`0.00–0.15` sim, `4.2–6.6 dB` HNR).
- **Not fixed by sampling:** argmax or `0.5 / top-25` on codebooks `8–15` or `1–15` does not recover speaker identity or HNR.
- **Already broken in codebooks 1–7:** decoding only codebooks `0–7` (`drive8.py`) keeps `0.30 / 14.5 dB` on the teacher, whereas the heads student drops to `0.10–0.11 / 6.7–8.1 dB`.
- **16 codebooks are still required for Vietnamese speech clarity:** Qwen3-ASR Vietnamese WER through Mimi is `8.9%` at 16 codebooks vs `15.7%` at 8 codebooks (English: `4.9%` vs `6.1%`; [vi_en_parity_plan](vi_en_parity_plan.md#7-two-way-speech)).

---

## 2. Why `ParallelHeads` Lost Voice

Four structural flaws in [`training/student/heads.py`](../training/student/heads.py) compound:

1. **No frame-local nonlinear RVQ computation:**
   Mimi uses Residual Vector Quantization (RVQ): codebook $j$ predicts the exact residual remaining after subtracting codebooks $0 \dots j{-}1$ of the *same frame*. With delays `min(j, 8)`, a single 2048-d trunk vector $h_t$ must simultaneously pack **9 different audio frames** ($t$ back to $t{-}8$, 720 ms of multi-frame state) and decode all 16 codebooks through single linear matrices `W_j @ (h_t + e_text)`. By contrast, a depformer gives each codebook dedicated nonlinear layers, per-step projections, and direct causal attention over $c_0 \dots c_{j-1}$ of the same frame.
2. **Misaligned text conditioning ([`heads.py`](../training/student/heads.py#L76)):**
   `self.depformer(transformer_out + self.text_emb(sequence[:, 0]))` adds **frame $t$'s** text embedding to codebooks `1..15`, which belong to frames $t{-}1 \dots t{-}8$ (up to 640 ms earlier).
3. **Distillation mismatch and RVQ marginalization:**
   The 3B teacher uses delays `[0, 0, 2, 2, ..., 2]`. Under the heads student's `min(j, 8)` delays:
   - On codebooks `3–7`, the student has read future frames the teacher has not; forward KL teaches it to blur over that context.
   - On codebooks `8–15`, the student has not read $8 \dots j{-}1$ of the same frame while the teacher has; KL forces the student to predict a marginal average over unseen residuals, destroying harmonic phase/structure (HNR `14.4 → 4.2–6.6 dB`, producing the raspy/breathy "old man" voice).
4. **Trunk capacity starvation:**
   Loss weighting gives codebook `0` weight `100` vs `1` for each acoustic codebook `1–15` (and GRPO optimizes text BLEU). The 14-layer trunk therefore spends its capacity on translation (+3.05 chrF over `student_dep10`), starving the acoustic linear heads.

---

## 3. Design Goals and A15 ANE Budget

What drove the heads student's wins was **not** the linear heads—it was:
- **12–14 main layers** (instead of 10),
- **16 codebooks** in and out,
- **ANE execution via Core ML** (sustained flat frame time with no GPU thermal throttling),
- **Distillation from cut + on-policy `both_z` + palettization QAT**.

The next student should **drop the delay pattern**, return to the **teacher's exact delays `[0, 0, 2, 2, ..., 2]`**, and replace the linear heads with an **ANE-friendly 16-codebook acoustic decoder**. Returning to exact teacher delays also:
- Cuts heard speech completion lag by **480 ms** (`160 ms` vs `640 ms`),
- Eliminates distillation delay mismatch across codebooks,
- Eliminates `teacher_leak` in on-policy distillation ([on-policy-distillation](agent-memory/on-policy-distillation.md#design)).

### A15 ANE Frame Budget (80 ms / frame at 12.5 Hz)

Measured on iPhone 13 Pro Max (A15, wired, 10 min flat, Mimi encode overlapped on GPU; [lm-npu](agent-memory/lm-npu.md#speed)):
- **Sustained ANE weight bandwidth:** `~17.2–17.5 GB/s` (14L 6-bit trunk `738 MB` = `42.6–44.3 ms`; 16 heads `51 MB` = `3.2–3.5 ms`). A small model costs more than one call suggests: 9 slices on the A15 cost `~1.8 ms` a call even at `512 × 1` (16.0 ms for 9 calls of `512 × 1`, 17.2 ms of `512 × 2`, 19.6 ms of `1024 × 1`, 28.0 ms of `1024 × 3`; [lm-npu](agent-memory/lm-npu.md#depformer-slices)). The estimates below use that measured cost per call.
- **Fixed non-LM costs on critical path:** Mimi decode on ANE `~6.9 ms` + host embedding gather/sampling `~4–5 ms` ≈ **`11–12 ms`**.
- **Core ML constraint:** Embedding gathers run on CPU in Core ML ([lm-npu](agent-memory/lm-npu.md#export)), so discrete token sampling and embedding lookup between sequential AR steps stay on the host.

Remaining ANE budget for the audio decoder (targeting `65–70 ms` total frame time to preserve a `10–15 ms` thermal/UI margin):

| Trunk Configuration | Trunk Weight Size | Est. A15 Trunk Time | Budget Left for Audio Head (for ≤ 68 ms total) |
|---|---:|---:|---:|
| 14L @ 6-bit (current) | 738 MB (measured) | 43 ms (measured) | **~13–14 ms** |
| 12L @ 6-bit | ~640 MB | ~37 ms | **~19–20 ms** |
| 14L @ 4-bit QAD | ~492 MB | ~29 ms | **~27–28 ms** |
| 12L @ 4-bit QAD | ~427 MB | ~25 ms | **~31–32 ms** |

### Orthogonal Speed Levers (Architecture-Independent)

1. **4-bit Palettization QAD (`--palettize-bits 4`):** Already implemented in [`training/student/palettize.py`](../training/student/palettize.py) ([student-palettize](agent-memory/student-palettize.md)). Shrinks a 14L trunk by `~246 MB` (`~14 ms` saved on A15 ANE). Must pass a quality gate (4-bit PTQ lost 4 chrF on the 3B in MLX; QAD has not been run yet).
2. **Trim the 48k text vocabulary to EN+VI pieces ([phone-student](agent-memory/phone-student.md#next)):** `text_linear` (`48,000 × 2048 = 98.3M` params, `74 MB` at 6-bit) costs `~4.3 ms/frame` on A15 ANE and `~196 MB` of host embeddings. Keeping only EN+VI pieces (~16–24k) saves `~2–2.5 ms/frame` with zero quality loss (teacher logits are indexed by kept IDs during distillation).

---

## 4. Candidate Architectures

### Candidate 1 (Recommended Primary): 12–14L Trunk + Compact Asymmetric Grouped Depformer (5–6 Steps)

Replace the heavy `d=1024` depformer with a **compact `d=512` (or `640`), 3-layer causal depformer** using **asymmetric RVQ grouping** and exact teacher delays `[0, 0, 2, 2, ..., 2]`:

```text
Step 0: [cb 0]          semantic, delay 0  (reads h_t + text_t)
Step 1: [cb 1]          1st acoustic RVQ   (strictly sequential, exact conditioning)
Step 2: [cb 2]          2nd acoustic RVQ   (strictly sequential, exact conditioning)
Step 3: [cb 3, 4]       2 cb in parallel   (both read exact 0..2)
Step 4: [cb 5, 6, 7]    3 cb in parallel   (all read exact 0..4)
Step 5: [cb 8..15]      8 cb in parallel   (all read exact 0..7)
```

*(Alternative 5-step grouping: `[0], [1], [2,3], [4..7], [8..15]`.)*

- **How steps connect:** At group step $g$, the input is `depformer_in[g](h_t) + sum_{q in group(g-1)} depformer_emb[q](c_q)` (mirroring how the main trunk already sums embeddings across codebooks). The 3 causal transformer layers attend over steps $0 \dots g$, and per-codebook linear heads predict all codebooks in group $g$ simultaneously.
- **Why asymmetric grouping:** Coarse acoustic codebooks (`cb 1, 2`) carry speaker identity, pitch contour, and harmonic structure and must not be marginalized. Residual energy decays exponentially with RVQ depth, so grouping `3–4`, `5–7`, and `8–15` preserves timbre while cutting sequential steps from 16 to 6.
- **Parameter & ANE math (`d=512`, 3L, SwiGLU hidden `1536`, per-step weights, 6-bit):**
  - 1 step (3 layers): `3 × (512×1536 + 512×512 + 3×512×1536) = 10.2M` params = **`7.7 MB`** at 6-bit.
  - 6 steps + 6 `depformer_in` (`2048×512`) + 16 output heads (`512×2048`): **`~85M` params (`~64 MB` total at 6-bit)**—smaller than a single main layer and ~5× smaller than `student_dep10`'s 436M-param depformer.
  - Projected A15 ANE time: `6 × ~1.9 ms` a call (measured per slice, above) ≈ **`11–13 ms`**, plus `~6 ms` of host for 6 samplings.
  - **Total A15 frame time:** **`~61–63 ms`** with a 14L 6-bit trunk (against the measured 67 ms of 9 slices of `512 × 2`), or **`~56 ms`** with a 12L 6-bit trunk.
- **Core ML export:** iOS 18 multifunction Core ML model (`6` entry points sharing a tiny `[3, 8, 6, 64]` KV state) or 6 small compiled models.
- **Training & RL:** Clean extension of `forward_depformer_training` in [`lm_training.py`](../training/finetune/lm_training.py#L180-L214); faster than a 16-step depformer; exact teacher delay alignment for off-policy distillation and on-policy `both_z`.
- **Risk:** Narrow `d=512` depformer trains from scratch (no warm start from 3B's `d=1024`). Mitigate with a 16k-update voice probe and higher acoustic weight (e.g., 3–5× on `cb 1–7`) or short depformer warmup if needed.

---

### Candidate 2 (Lowest Quality Risk): 12L Trunk + Warm-Started 9-Step Hybrid Depformer (`0–7` AR + `8–15` Parallel)

Cut directly from the 3B teacher's own `d=1024` depformer with **2 layers** (e.g., layers `[0, 5]`), keeping all 9 weight slices `0..8`:
- **Steps 0–7:** exact 1-by-1 AR on `cb 0..7` (proven to keep `0.32` speaker sim / `12.2 dB` HNR in `student_dep10`).
- **Step 8:** 1 parallel step through slice `8` (which the 3B already shares across `8–15`), outputting `cb 8..15` via the 3B's own `linears[8..15]` and `depformer_norms[8..15]`.
- **Pros:** 100% warm-started from the 3B teacher (preserves voice from step 0); minimal changes to [`cut.py`](../training/student/cut.py) and [`lm_training.py`](../training/finetune/lm_training.py); zero grouping risk on `cb 0–7`.
- **ANE math (`d=1024`, 2L, 6-bit):**
  - 9 slices × `25.2 MB` + `depformer_in`/`linears` (`~40 MB`) = **`~267 MB`** → `15.5 ms` DRAM + `4.5 ms` dispatch ≈ **`~20 ms`**.
  - Total A15 frame time: **`~69 ms`** with a 12L 6-bit trunk (tight margin at 6-bit; drops to **`~52 ms`** if paired with 4-bit QAD on trunk + depformer).
  - *(3 layers at `d=1024`, 6-bit is `~380 MB` / `~27 ms`, which exceeds the 6-bit budget on a 12L trunk and requires 4-bit QAD or a 10L trunk.)*

---

### Candidate 3 (Zero Parallel Approximation): 12L Trunk + Shared-Weight Narrow Full-AR Depformer (`d=512`, 3L, 16 Steps)

Same `d=512`, 3-layer depformer as Candidate 1, but with **`weights_per_step = False`** and **exact 16-step causal AR** across all 16 codebooks (`cb 0 → 1 → ... → 15`):
- **Pros:** Zero parallel marginalization anywhere (exact teacher KL match on all 16 codebooks, highest voice quality ceiling); only `~35M` depformer parameters total; a single **`7.7 MB`** Core ML model that may stay resident in A15 ANE SRAM/SLC after step 0.
- **Cons:** 16 sequential host↔ANE calls per frame: at the measured `~1.8 ms` a call, `~29 ms` of calls alone, plus `~10 ms` of host for 16 samplings, so `~76 ms` a frame on a 12L trunk. It misses the 65–70 ms budget unless the calls get cheaper.

---

### Candidate 4 (Longer-Term Research Bet): 14L Trunk + Continuous Latent Flow-Matching Head

Keep `cb 0` discrete (semantic), and replace `cb 1..15` discrete heads with a compact **2–4 step Flow-Matching / Consistency MLP-ResNet** (~30–50M params) predicting Mimi's 512-d continuous quantized latent $\hat{z}_t$ (delay 2) directly into Mimi's decoder (re-quantizing $\hat{z}_t$ on CPU via Mimi's RVQ or linearly projecting $\hat{z}_t$ into trunk history).
- **Pros:** **Single Core ML ANE call** (`~4–6 ms`, all Euler steps fused inside one graph with no host round trips); continuous 32-cb-equivalent audio fidelity and HNR; leaves ANE budget for 14–16 trunk layers.
- **Cons:** Cannot apply `token_kl` directly to audio (requires flow-matching velocity loss on continuous latents while text + `cb 0` keep `token_kl`); cannot warm-start from the 3B's depformer; requires re-exporting Mimi's decoder to take continuous $\hat{z}_t$ and building a new training/inference path.

---

## 5. Comparison Summary

All A15 numbers below assume Core ML on ANE, Mimi decode on ANE (`~7 ms`), host (`~4–5 ms`), and Mimi encode overlapped on GPU (`0 ms` critical path).

| Candidate | Trunk (6-bit) | Audio Head (6-bit) | ANE Calls | Speech Lag | Est. A15 Frame | Warm-Start | Voice Risk |
|---|---|---|---:|---:|---:|---|---|
| Current `heads14_cb16` | 14L (43 ms) | 16 linear (3.5 ms) | 1 | 640 ms | **53 ms** (measured) | Trunk only | **Failed** (0.10 sim, 6 dB HNR) |
| **1. Grouped Thin (`d=512`, 3L)** | 14L (43 ms) / 12L (37 ms) | 6-step grouped (~12 ms) | 6 | **160 ms** | **~62 / ~56 ms** | Trunk only | **Low** (verify at 16k) |
| **2. Hybrid Warm-Start (`d=1024`, 2L)** | 12L (37 ms) | 0–7 AR + 8–15 par (~20 ms) | 9 | **160 ms** | **~69 ms** (~52 @ 4-bit) | **Full** | **Lowest** |
| **3. Shared Narrow AR (`d=512`, 3L)** | 12L (37 ms) | 16-step shared (~29 ms of calls) | 16 | **160 ms** | **~76 ms** (misses) | Trunk only | **Lowest** |
| **4. Flow-Matching Head** | 14L (43 ms) | 4-step flow (~5 ms) | **1** | **160 ms** | **~59 ms** | Trunk only | Medium (new recipe) |

---

## 6. Execution and De-Risking Plan

Follow the project invariant of profiling and gating before full runs ([CONTEXT.md](../CONTEXT.md#essential-invariants)):

### Step 1: A15 ANE Micro-Benchmark (1–2 hours, zero training)
Export dummy Core ML models on the Mac and run 10-minute sustained benchmarks on the iPhone 13 Pro Max beside 12L and 14L trunks (both 6-bit and 4-bit):
- Candidate 1: 6-step grouped `d=512`, 3L (`~64 MB` total),
- Candidate 2: 9-step `d=1024`, 2L (`~267 MB` total),
- Candidate 3: 16 calls of one shared `d=512`, 3L model (`7.7 MB`) to measure true Core ML per-call dispatch latency and whether A15 ANE keeps `7.7 MB` warm in SRAM.

This establishes exact A15 frame times and decides whether 14L or 12L fits comfortably under `65–70 ms`.

### Step 2: Add an Automated Voice + Speech Gate
Done: since `voice_scores.py` ([voice-scores](agent-memory/voice-scores.md)), every validation and `eval.py` pass reports each set's `mean_speaker_sim` and `mean_hnr_db`. The scores are reported, not gated. The floors below are mic-clip numbers. Validation's sets score lower for every model: on FLEURS-vi dev the heads release reads 0.21 / 3.2 dB and the depformer student 0.27 / 4.7 dB. So compare a candidate with those models on the same set, not with a floor:
- **Speaker similarity:** WeSpeaker ResNet34 cosine similarity on the 5 Vietnamese mic clips (pass floor: `≥ 0.30`).
- **Harmonic quality:** Harmonics-to-noise ratio (pass floor: `≥ 12.0 dB`).
- **Speech intelligibility (EN→VI):** Qwen3-ASR WER on a small FLEURS-en slice to confirm 16-codebook Vietnamese clarity holds.

### Step 3: 16k-Update Voice & chrF Probe (~3 hours on GPU)
Because `student_heads14_cb16` already showed its voice collapse at update 16,000, run a 16k-update off-policy distillation probe on the best ANE-viable candidate(s):
- Start with **Candidate 1** (6-step grouped `d=512`, 3L). If `cb 3–7` grouping softens HNR or speaker sim below the `0.30 / 12 dB` floor, test **Candidate 2** (9-step warm-started `d=1024`, 2L) or **Candidate 3** (16-step shared AR, if Step 1 shows it fits the A15 budget).
- Also test whether upweighting `cb 1–7` (e.g., 3–5× relative to `cb 8–15`) during early distillation accelerates voice convergence for a fresh depformer.

### Step 4: Full Training, On-Policy RL, and Palettization QAT
Once a candidate clears the 16k voice + chrF gate and the A15 sustained-speed gate:
1. Full off-policy distillation from the latest 3B teacher (~190k updates, checking voice gate at 16k, 64k, 128k, final).
2. On-policy distillation + GRPO (`both_z`, z-scored), verifying no voice degradation after RL.
3. Palettization-aware distillation (`--palettize-bits 6`, and optionally `--palettize-bits 4` if quality holds for extra A15 headroom / 14L trunk).
4. Core ML ANE export, full 13-set text + voice gate, and 10-minute sustained iPhone test.

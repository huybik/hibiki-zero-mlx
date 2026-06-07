# Plan: LoRA-on-main fine-tune, Mimi-cached — 10h "test the water"

Goal: **validate the training mechanics** on the Mac for adapting hibiki-zero to a
**new source language (Vietnamese → EN)** using ~10h of data. Quality is secondary; the
output of this run is a yes/no on "does a LoRA-on-main-only fine-tune (depformer/Mimi/
embeddings frozen) move the loss for a new language."

Source tricks being tested: pre-tokenize Mimi once, bf16 LoRA (not full FT), LoRA only the
main transformer, length-bucketing, subset-first (10h), gradient checkpointing.

---

## 0. Key decisions (locked, with rationale)

**Framework = PyTorch/MPS, not MLX.** Training needs autograd + LoRA + a training forward.
The PyTorch `moshi` package already ships all three (`LMModel.forward` → `LMOutput`,
`modules/lora.py`, `Mimi.encode`). The MLX fork is **inference-only** — using it would mean
hand-writing the whole training loss/backward. So the MLX work stays untouched; this is a
separate PyTorch stack living in `finetune/`. (Same split the repo already makes between the
MPS `serve/generate` stack and the MLX path.)

**Freeze map** (the three tricks, mapped to real modules in `moshi/models/lm.py`):

| Module | Action | Why |
|---|---|---|
| Mimi codec | **not loaded in the loop** (cached) | removes the ~58%/frame codec cost entirely — biggest win |
| `emb`, `text_emb`, `depformer_emb`, `depformer_text_emb` | `requires_grad=False` | audio/text token embeddings are language-agnostic |
| `depformer`, `depformer_in`, `depformer_norms`, `linears` | `requires_grad=False` | acoustic head; a new *source* lang adapts the main, not the head |
| `out_norm`, `text_linear` | freeze (output is still English) | nothing about the EN text head changes |
| `transformer` (the 28-layer main) | **LoRA only** via `replace_all_linear_with_lora(model.transformer, rank, scaling)` | the one thing that must adapt to a new source language |

Audio loss still backprops *through* the frozen depformer into the main via LoRA — that's
desired, not wasted.

**This contradicts the repo's own `docs/distill_plan.md` Track A** (which says adding a
language unfreezes the full 3B + AR depformer, upstream on GPU). That's intentional — this
LoRA-main-only, depformer-frozen, on-device bet is precisely the cheap hypothesis worth
*testing* before committing to a full upstream fine-tune.

## 1. Concrete model facts (from `weights/config.json`)

- `K = 33` codebooks total: **1 text + 32 audio**. The 32 audio = **16 target/EN (predicted,
  `dep_q=16`) + 16 source (input-only)**. `card=2048` audio, `text_card=48000`.
- `LMModel.forward(codes[B,33,T])` handles delays/interleave internally and returns masked
  `logits [B,16,T,2048]` + `text_logits [B,1,T,48000]` — **ready for cross-entropy**, no
  manual shifting.
- **#1 implementation detail to verify first:** the exact row order of the 33 in `codes`
  (text / EN-audio-16 / source-audio-16) and the `audio_offset`. Confirm against how
  `inference.py` / `LMGen` assembles the live input before trusting the layout.

## 2. De-risk: prove the loop *before* the data pipeline

Two independent things can break — the **trainer** and the **Vietnamese data synthesis**.
Don't debug them together. So **Phase A first, on an existing language**, with self-generated,
guaranteed-correct targets. Cheap, and catches every trainer bug (freezing, codes layout,
loss masking, MPS dtype) with zero external data risk. Then Phase B swaps in real Vietnamese.

## 3. Phases & deliverables (all in `finetune/`)

**Phase A — trainer smoke test (existing lang, ~1h of FR/ES)**
- `cache_codes.py` — Mimi-encode source + EN audio for each clip → save `codes [33,T]` int16
  shards (`.npy`/`.safetensors`) + aligned EN text tokens. *This is the Mimi-caching trick.*
  Run once, offline.
- `train_lora.py` — load LM, apply freeze map + `replace_all_linear_with_lora(model.transformer,
  r=16, scaling=2.0)`, Adam on LoRA params only, CE loss from `LMOutput` masks. bf16 forward,
  fp32 LoRA/optimizer.
- **Gate:** *overfit a single batch to ~0 loss.* If it can't, the trainer is wrong — stop and
  fix. Confirms layout + masking + grad flow.

**Phase B — Vietnamese data (10h)** — see §7 for sourcing
- `build_pairs.py` — take a 10h slice of **VietBud500** (transcripts ship with it → skip Whisper)
  → MADLAD/NLLB **Vi→EN** text → EN TTS audio (speaker-ish) → **concatenate consecutive clips
  into 30–75s chunks with artificial silences** (VietBud500 clips avg ~2.8s; coarse alignment per
  distill_plan §B.2). Output: `(vi_audio, en_text, en_audio)` triples.
- Feed triples through `cache_codes.py` → cached `codes` shards. **10h ≈ 450k frames @ 12.5 Hz.**
- Length-bucket the shards (utterances 30–75s) — the cheap 10–30% throughput trick.

**Phase C — train + read the result**
- Run `train_lora.py` on the 10h cache, a few epochs.
- **Success = mechanics validated:** (1) train loss decreases steadily; (2) **silence-in test
  passes** (zeros → rms<0.10, peak<1.1 — the repo's standard babble/clip detector); (3) it fits
  in 48GB; (4) a held-out Vietnamese clip produces *non-garbage* EN text after merge.
- Merge LoRA (`replace_lora_with_linear`) → bf16 checkpoint → existing `scripts/convert_mlx_q4.py`
  → listen via the MLX fast path. (No ASR-BLEU gate yet — that's for the "move real quality"
  goal, not mechanics.)

## 4. Feasibility on M4 Pro 48GB (rough)

- Resident: full LM bf16 ≈ **5.8GB** (all frozen); LoRA params + Adam fp32 ≈ <0.2GB; Mimi **not
  loaded**. Activations at T≈940, B=1: logits dominate (~0.4GB), layer activations ~1–2GB.
  **Comfortable in 48GB** — gradient checkpointing is the *optional* knob if you push batch
  size, not required.
- Speed: caching removes the 58% codec cost. Training fwd+bwd ≈ a few ×10⁻² s/frame on MPS →
  order **~hours/epoch for 10h**. Fine for a water test; this is why you start at 10h, not 100h
  (subset-first / curriculum trick).

## 5. Tricks coverage check

✅ Mimi pre-tokenize once (`cache_codes.py`) · ✅ bf16 LoRA not full FT · ✅ LoRA only the main
transformer · ✅ length-bucket · ✅ subset-first (10h) · ✅ gradient checkpointing (available,
optional). All six accounted for.

## 7. Vietnamese data sourcing (researched)

**There is no off-the-shelf Vietnamese→English speech-translation corpus.** You build it from
monolingual Vietnamese audio + synthesized targets — exactly the distill_plan Track A recipe.

Why nothing ready-made fits:
- **`kyutai/Audio-NTREX-4L`** is speech-to-**text** *eval only* (FR/ES/PT/DE→EN). Columns:
  `source_audio` (ElevenLabs **TTS**, ~4–72s) + `source_text` + aligned transcript + `target_text`
  (**English text only — no target audio**). 3,600 rows (1,800 valid + 1,800 test). No Vietnamese,
  and not training data. Note: even this set has no EN target *audio* — hibiki being S2ST, you must
  TTS the English target into audio for **training** regardless of source.
- **CoVoST-2** (`facebook/covost2`): 21 langs → EN, but **Vietnamese is not included**.
- **PhoST** (arXiv 2208.04243; 508h, 331K triplets): **English→Vietnamese** — wrong direction.

What to actually use:

| Role | Source | Scale / notes |
|---|---|---|
| **Training (bulk source)** | **VietBud500** (`linhtran92/viet_bud500`) | ~500h, 634k clips, 16kHz, **transcripts included** (skip Whisper). **CC-BY-NC-SA-4.0 = same license as the model.** Spontaneous (podcast/travel/food). Clips avg ~2.8s → **concatenate to 30–75s** chunks. |
| Training (extra) | **Common Voice `vi`** | tens of h, real read speech, accent variety. |
| **Held-out eval** | **Build "Audio-NTREX-VI"** | **NTREX-128 includes Vietnamese** (line-aligned to English, same newstest2019 source; CC-BY-SA-4.0). TTS the Vi reference lines → Vi source audio; the English line is `target_text`. Same construction kyutai used for 4L → directly comparable S2T eval. |

For the 10h water test: a 10h VietBud500 slice → MADLAD/NLLB Vi→EN → EN TTS → `cache_codes.py`.
No Whisper needed (transcripts ship with VietBud500).

## 8. Top risks

- **Hypothesis itself may fail:** if depformer-frozen LoRA-on-main *can't* learn Vietnamese
  (distill_plan bets it can't), Phase C loss stalls → that's a *valid, cheap* answer pointing to
  "unfreeze depformer / go upstream." Don't over-invest before Phase C reads out.
- **Codes layout / `audio_offset`** wrong → silent garbage. Phase A's overfit-a-batch gate
  catches it.
- **Vi label noise** (Whisper-VI, MADLAD Vi→EN) — acceptable for mechanics; matters only for the
  later quality goal.
- **MPS bf16 op gaps** — fall back to fp16/fp32 on the offending op (the repo already hit this;
  `--bf16` is opt-in for that reason).

---

This is ~3 small scripts in `finetune/`, no edits to the inference stacks.

**Next step:** Phase A — confirm the exact `codes` row layout from `inference.py`, then write
`cache_codes.py` + `train_lora.py` and run the overfit-a-batch gate.

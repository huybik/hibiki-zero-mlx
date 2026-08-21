# VI→EN parity plan

Goal: make Vietnamese→English work as well as the upstream Hibiki-Zero source
languages (FR/ES/PT/DE) on translation quality, latency behavior, voice
preservation, and free-running generation health.

This plan combines:

- the measured failures in `docs/analysis/validation_collapse_analysis.md`,
  `docs/analysis/step135k-check.md`, and the objective in
  `docs/analysis/loss_function.md`;
- the Hibiki-Zero training recipe (arXiv 2602.11072, "Simultaneous Speech-to-
  Speech Translation Without Aligned Data");
- the new Common Voice based data pipeline described below.

## 1. Where we are

Evidence from the direct VI-EN run `grounded_v2_full_direct_voice_5epoch`:

| Signal | Result | Meaning |
| --- | --- | --- |
| Unseen PhoMT teacher-forced loss | 3.68 → 3.13 (step 18k → 135k), still falling | Translation knowledge is learnable but not saturated; 683k rows are not yet too many |
| FLEURS teacher-forced loss | 5.84 → 7.28, rising 13 checkpoints | Domain conflict + only 1,392 unique FLEURS rows reused ~77× |
| Free-running on PhoMT holdout (step 135k) | BLEU 2.07, chrF 18.85, WER 159%, EOS 4/6 | Teacher-forced loss does not transfer to free-running; exposure gap dominates |
| Free-running on FLEURS (step 18k) | BLEU 0.88, chrF 17.12, but repetition gate failed | Best checkpoint already unhealthy at generation |

Diagnosis, ranked:

1. **Data quantity/variation**: ~1,114 h of synthetic PhoMT VI speech (683k
   rows) and 1,392 unique FLEURS-domain rows versus the paper's 40,000 h per
   source language of real multi-speaker audio (~36× less audio, and far less
   speaker variation).
2. **Exposure gap**: supervised training is fully teacher-forced (gold English
   text+audio history in input); inference is free-running. No training stage
   optimizes the free-running regime, so repetition, missing EOS, and error
   compounding are expected.
3. **Voice-preservation mismatch**: audio CE is applied to pairs whose English
   target speaker is not verified to match the Vietnamese source speaker,
   teaching speaker conversion instead of voice preservation.
4. **Stopping/selection**: five fixed epochs at fixed LR selected checkpoints
   against a mismatched 138-row FLEURS validation set.

## 2. What the Hibiki-Zero paper does

Five stages (§4.2 of the paper):

1. **Text backbone init**: Helium-1-2B LLM weights.
2. **Audio pretraining**: single-stream multilingual audio, 1M steps,
   batch 144.
3. **Coarse ST training**: teacher-forced CE on 40,000 h per source language
   (~4M utterances), 500k steps, batch 96, loss on both streams, source noise
   augmentation. Targets are delayed by random per-sentence silences
   δ_i ~ U(0, δ·d_i) plus punctuation pauses U(0, μ) to teach simultaneous
   behavior. Same exposure gap as our SFT — but at ~36× the audio volume, all
   real multi-speaker speech.
4. **Fine-tuning on natural-pause TTS data**: <200 h synthetic targets where a
   synced-stream TTS regenerates the English audio with natural pauses,
   **voice-transferred onto the source speaker**. 1K steps, batch 16,
   LR 1e-6. Then distills into a lighter self-copy (20K updates). This stage
   fixes the unnatural spliced-silence targets and is the paper's structural
   answer to voice preservation.
5. **GRPO reinforcement learning**: G=4 free-running generations per input,
   scored by process rewards r_t = (1−α)·BLEU(partial at frame t) +
   α·BLEU(full output), α=0.4, rewards every 8 input words, batch 32,
   LR 2e-7, 2000 updates, τ=20, temperature 0.8 / top-k 250. This stage trains
   directly in the free-running regime and is what teaches EOS, length
   control, and recovery from the model's own errors.

Key paper finding: RL could not teach behaviors the base model "was never
trained in that manner during supervised training" — the supervised stages
must already leave exploration room (δ_i < d_i), and RL then sharpens the
policy.

## 3. Gap analysis: paper vs. this repo

| Dimension | Paper | This repo | Action |
| --- | --- | --- | --- |
| Supervised ST data | 40,000 h / language (real, multi-speaker) | ~1,114 h PhoMT (synthetic) + 1,392 FLEURS rows | Phase 1 |
| Target-domain speech variation | Massive, multi-speaker | 1,392 unique rows | Phase 1 |
| Voice-matched targets | TTS voice transfer from source speaker (stage 4) | Timbre flag on ~51% of PhoMT rows, unverified | Phases 1+3 |
| Natural simultaneity targets | Natural-pause TTS (stage 4) | CTC-timed alignment only | Phase 3 (lightweight analog) |
| Free-running training | GRPO with BLEU process rewards (stage 5) | None | Phase 4 |
| Early stopping | Best quality/latency checkpoint | 5 fixed epochs, teacher-forced selection | Phase 2 |

## 4. New data: Common Voice pipeline

Source: Mozilla Common Voice **English** (scripted, CC0): one validated
sentence per clip, tens of thousands of speakers, thousands of validated
hours.

Pipeline per selected clip:

1. Take the validated English clip: real English audio + gold English text
   (this becomes the English target stream, unchanged).
2. Translate the English sentence to Vietnamese with Gemini.
3. Generate Vietnamese speech from the translated text with a TTS **conditioned
   on the clip's original speaker** (voice transfer), producing a timbre-matched
   Vietnamese source audio.
4. Cache as a grounded-v2 style row: Vietnamese source Mimi codes (with codec
   EOS), CTC-timed English text (with EOS), English target Mimi codes.

Why this direction is right: the model needs Vietnamese *source* audio and
English *target* audio/text. Common Voice gives real English targets and, via
voice transfer, Vietnamese sources that share the target's speaker identity —
every row is verifiably timbre-matched, which fixes the audio-CE supervision
boundary from the analysis doc.

### Sizing

Total goal: **~40,000 h of Vietnamese source speech**, matching the paper's
per-source-language training volume. The existing cache contributes ~1,114 h,
so the new pipelines must produce roughly 39,000 h more.

Common Voice English validated audio alone is only a few thousand hours, so
the Section 4 pipeline (EN clip → Gemini translate → timbre-matched VI TTS)
is applied across multiple scripted English corpora, for example:

- Mozilla Common Voice (English, CC0) — thousands of speakers, short clips;
- LibriSpeech / MLS English (read audiobooks, CC-permissive);
- GigaSpeech / SPGISpeech or similar large licensed read corpora.

| Segment | Now | Target | Purpose |
| --- | ---: | ---: | --- |
| Scripted-EN derived rows (CV first, then other corpora) | 0 | **scaled until total VI ≈ 40k h** | Bulk VI-source/EN-target with verified timbre match; massive speaker variation |
| PhoMT rows (existing) | 683,164 (~1,114 h) | keep | Translation breadth, human-aligned text pairs |
| FLEURS rows (existing, real VI speech) | 1,392 | keep, **single pass — no reuse within an epoch** | Real-speech anchor; validation domain |
| Validation | 138 FLEURS rows | + stratified holdouts from each new corpus + 1,068-row PhoMT holdout | Per-domain selection |

At ~7 s average clip length, 40k h is on the order of 20M clips; prioritize
corpora by speaker count and audio quality, and generate in tranches (e.g.
validate the pipeline on 1k h before scaling).

### Pipeline quality gates (before any training)

- **Translation**: back-translate spot check on a random 1k sample; require
  ≥95% accept rate; drop rows where Gemini output is empty, identical to
  English, or longer than 2× the source.
- **TTS**: ASR the generated Vietnamese audio on a random 1k sample; require
  WER < 15% against the translated text; drop clips with clipping, silence
  >50%, or duration >280 frames (cache cap).
- **Timbre**: verify speaker-embedding similarity between generated VI audio
  and the original EN clip; keep only pairs above the same threshold used for
  PhoMT `cross_lingual_timbre_matched`.
- **Dedupe**: near-duplicate sentence text within Common Voice is common —
  dedupe by normalized text before TTS to avoid wasting generation budget.
- **Split**: hold out ~2k rows (never trained) as the CV validation manifest,
  stratified by speaker and duration.

Known risk: the Vietnamese source side becomes 100% synthetic. Mitigation is
to keep real FLEURS rows in the mixture and to monitor the FLEURS validation
loss separately (never merge domain losses into one number).

## 5. Phases

Run phases strictly sequentially; commit and verify exit gates before starting
the next phase.

### Phase 1 — Data

1. Build the scripted-EN pipeline (translate → TTS voice-transfer → cache),
   starting with Common Voice.
2. Produce new grounded rows passing all Section 4 gates, scaled until total
   Vietnamese source audio ≈ 40k h (tranche 1: 1k h pipeline validation).
3. Freeze new manifests: train mixture ≈ PhoMT + scripted-EN rows + FLEURS
   (single pass, no reuse), plus separate validation manifests (FLEURS,
   PhoMT holdout, per-corpus holdouts).
4. Rebalance the mixture so the read-speech domain is no longer a 5% sliver;
   target roughly 50/50 between PhoMT-style and read-speech data, tunable in
   Phase 2.
5. Log unique-row counts and per-row reuse explicitly in the receipt; assert
   FLEURS reuse = 1× in the manifest validator.

Exit gate: caches pass the smoke/resume check; validation manifests are
row-disjoint from training; mixture ratios are recorded in the receipt.

### Phase 2 — Supervised SFT (fixed recipe)

1. Initialize from upstream Hibiki-Zero as before.
2. Batch 16, 280-frame cap, fused AdamW (0.9, 0.95), wd 0.1 — unchanged.
   Note: at ~40k h the corpus no longer fits the old 27k–36k step budget
   (~400k steps would be one pass at batch 16); set the step cap and LR
   schedule after measuring throughput, and keep multi-epoch replay off the
   table for FLEURS.
3. Cap the first run at a measured fraction of one epoch (start with the
   equivalent of the old 27k–36k step budget, then extend while validation
   improves); no five-epoch replay.
4. Validate every 3,000 steps on all three manifests; report losses
   separately; stop after two consecutive regressions on the primary
   deployment domain.
5. LR: repeat once at fixed 1e-6 to confirm the early optimum, then compare a
   decayed variant (1e-6 → 1e-7 before step 27k) in a separate run. Change
   only one variable per run.
6. Promotion gate = teacher-forced losses (per domain, no merging) **plus**
   the deterministic correct-source free-running check on the same 128-row
   subset: nonempty ≥122/128, EOS ≥116/128, repeated-4-gram ≤12/128, length
   ratio ≤2.0, no material chrF regression.

Exit gate: a checkpoint that passes the full promotion gate, with FLEURS and
CV validation losses no longer diverging from training loss.

### Phase 3 — Voice-matched fine-tune (stage-4 analog)

Lightweight version of the paper's natural-pause fine-tune, without training
a synced-stream TTS:

1. Restrict audio CE to rows with verified timbre match (all CV rows; PhoMT
   timbre-matched rows after verification). Unmatched PhoMT rows keep text CE
   only.
2. Fine-tune the Phase 2 checkpoint on the timbre-matched subset only:
   ~1–2k steps, batch 16, LR 1e-6 (mirrors the paper's stage-4 budget).
3. Add speaker-embedding similarity (source VI vs generated EN) to the eval
   harness alongside English ASR content accuracy.

Exit gate: speaker similarity improves or holds on the CV holdout while
teacher-forced losses and free-running gates hold on FLEURS/PhoMT.

### Phase 4 — GRPO reinforcement learning (stage-5 analog)

1. Start from the Phase 3 checkpoint. Data: sentence-level aligned rows with
   reliable references (CV rows are ideal: gold EN text; PhoMT rows usable).
2. Sample G=4 free-running generations per input (temperature 0.8, top-k 250,
   both streams — same decoding as `finetune/eval.py`).
3. Process reward: r_t = (1−α)·BLEU(partial text at t, reference prefix) +
   α·BLEU(full text, reference), α=0.4, rewards every 8 input words.
4. GRPO without KL regularization; batch 32, group 4, LR 2e-7, start with
   ~500–1000 updates (paper used 2000; scale to measured stability), evaluate
   every 10·τ updates, select by quality/latency trade-off.
5. Keep the free-running promotion gates from Phase 2 as hard filters on any
   promoted checkpoint.

Exit gate: BLEU/chrF improve on all three holdouts without worse EOS,
repetition, or length ratio, and speaker similarity holds.

### Sequencing rationale

- Phase 1 first: RL on a weak base rewards noise; the paper's RL worked
  because the SFT base was already strong. Data is also the longest-lead item.
- Phase 2 fixes stopping/selection so we stop confusing "more training" with
  "better model" and get a clean base for later phases.
- Phase 3 is cheap and directly targets voice preservation before RL locks in
  a policy.
- Phase 4 attacks the exposure gap in the regime the model actually runs in —
  this is the paper's answer to exactly the repetition/EOS failures measured
  in `step135k-check.md`.

## 6. Success criteria (parity with upstream languages)

Measured on the three held-out manifests with the same decoding config as the
paper's inference (temperature 0.8, top-k 250):

1. Teacher-forced: content and audio losses on every domain stop diverging
   from training loss across the whole run.
2. Free-running: chrF and BLEU on VI→EN within reach of the paper's reported
   per-language spread; EOS found ≥90%; repeated-4-gram outputs ≤10%; mean
   length ratio ≤2.0.
3. Voice: speaker-embedding similarity between Vietnamese source and generated
   English comparable to the paper's cross-lingual speaker-similarity results.
4. Latency: translation begins before source end (simultaneous behavior), not
   full-utterance wait.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Synthetic VI source audio creates an acoustic domain gap vs real speech | 40k h target is met with synthetic sources; keep the 1,392 real FLEURS rows in-mixture (single pass); monitor FLEURS validation separately; consider real-VI data acquisition if FLEURS loss stalls |
| Gemini translation noise propagates into training | Back-translation spot checks; drop suspicious rows; keep PhoMT (human-aligned) as translation anchor |
| TTS voice transfer imperfect on some CV speakers | Speaker-embedding threshold gate; drop failing speakers rather than rows silently |
| GRPO compute cost on a 3B multistream model | Small update budget first (500–1000); G=4 as in paper; reward computed on text stream only |
| RL destabilizes the Phase 2/3 gains | LR 2e-7, hard promotion gates, keep best pre-RL checkpoint as fallback artifact |

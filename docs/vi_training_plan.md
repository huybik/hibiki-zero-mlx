# Adding Vietnamese (vi→en) — training plan

How the paper adds a new language, where our current run stands, and the revised recipe.
Companion to `finetune.md` (mechanics of the finetune stack) — this doc is the *strategy*.

## Paper method (Hibiki-Zero §4.6 + §4.2.3, arXiv 2602.11072v1)

Adding a language = **supervised finetuning on a coarse-aligned ST dataset**, then distill + RL downstream:

1. **Data — <1000 h (850 h for Italian).** Source = real speech (Whisper-transcribed), target text = MADLAD translation, **target audio = TTS with speaker conditioning** from the source utterance.
2. **SFT** — full-model finetune from the base multilingual checkpoint, **CE on both text and audio streams** + source-noise augmentation, ~1K steps at **batch 16**.
3. **Coarse alignment / streaming delay** — insert silence in the target so sentence *i* starts `δᵢ ~ U(0, δ·dᵢ)` after the source sentence start (**δ=0.5**), plus `U(0,μ)` pauses at punctuation (**μ=2**). The randomized delay *is* how streaming latency is taught.
4. **Downstream (later, not our concern yet)** — distill to a smaller copy (20K steps) → GRPO RL with a BLEU process reward (2K steps, batch 32, lr 2e-7).

Our *recipe* matches 1–3 (SFT + CE text/audio + randomized per-clip delay). Two gaps remain: **scale** (§Where we stand) and **method** — the paper does a **full-model finetune**; our runs so far used **LoRA r32**. See §Full-model, not LoRA.

## Where we stand (run `finetune/runs/vn_phomt_combined`)

- Crashed by BSOD at **step 605 ≈ end of epoch 1** (609 steps/epoch @ eff-batch 4). Latest resumable checkpoint: **step 500**. Cause: `sort_by_length=true` puts the longest clips (up to 51 s) in the final batches → MPS working set 8.6→37.7 GB (over the 36 GB cap) → kernel panic. **Fixed:** `--max-frames` drops over-long cached samples (see `finetune.md`); use **280 (22 s)** — drops 3.2 %, peak batch well under the crash point.
- **`adapter_step000500` does not translate.** val128 greedy → mostly empty outputs + degenerate English loops ("the idea of the idea of…", "the 1000th anniversary of the 17th anniversary of…"), not conditioned on the VI source. This is the textbook data-starvation symptom.

**Root cause: ~7.8 h of vi data vs the paper's 850 h floor (~100× short).**

| | Paper (Italian) | Us (vi) |
|---|---|---|
| coarse-aligned data | **850 h** | **~7.8 h** (1.64 PhoMT + ~6 FLEURS) |
| source speech | real, Whisper-transcribed | FLEURS real + PhoMT synthetic |
| target audio | TTS, speaker-cond, randomized delay | Kokoro TTS, randomized delay ✓ |
| SFT | full model, batch 16 | LoRA r32, eff-batch 4 → **switch to full model** (§Full-model, not LoRA) |

No schedule/LR change fixes a 100× data gap.

## Full-model, not LoRA

The one place our setup diverges from the paper is the SFT method: the paper does a **full-model finetune** from the base checkpoint; we ran **LoRA r32**. Adding a *new source language* is the worst case for a low-rank adapter — the base model has never seen Vietnamese input, so this is a high-rank change to the input representation, not the style/domain nudge LoRA is good at. There is no evidence LoRA r32 has the capacity for it; the assumption is untested and the paper deliberately chose full finetune.

So for the real scaled run: **full-model SFT, matching the paper.** The cost objection (full 3B + Adam ≈ 36 GB, won't fit the 36 GB MPS cap — why we used LoRA locally) disappears on CUDA, where full finetune at batch 16 is routine. Keeping LoRA on the big data run would only reintroduce an unvalidated capacity confound: if it underperformed, we couldn't tell data from method.

**Code:** `train_lora.py --full-finetune` unfreezes every LM param, uses a single param group on `--lr`/`--lr-schedule`, and saves `model_step*.safetensors` (metadata `target=full`); `eval_lora.py`/`validate_lora.py` auto-detect and load full checkpoints without LoRA insertion.

**The plumbing test stays LoRA on MPS** (§Immediate plumbing check) — it's a cheap local capacity/soundness probe, not the shipping run, so the LoRA↔full mismatch there is acceptable. The scaled CUDA run is the one that goes full-model.

## Revised recipe — keep SFT, fix the inputs

**Track 1 — Scale the data (the fix that matters).** The **full PhoMT corpus is ~2.9M en–vi text pairs; we synthesized only 992.** Drive the existing Kokoro pipeline (`training-data/`) to synthesize **50–150 h**. Mix in **real vi source** (CommonVoice-vi 100s of h + VIVOS + FLEURS) transcribed+translated per the paper so the final SFT slice isn't 100 % TTS-source (TTS-vi → real-vi domain gap). Target: low-hundreds of hours.

**Track 2 — Paper-faithful alignment aug (cheap, `cache_codes.py`).** Re-sample the target delay per epoch (or pre-bake N variants) instead of deterministic-per-id, and add μ=2 punctuation pauses. Exact §4.2.3 recipe; helps latency robustness.

**Track 3 — Training.** **Full-model SFT** (`--full-finetune`, §Full-model, not LoRA) + CE(text 5 : audio 1); `--max-frames 280`; more epochs; batch 16 (grad-accum as needed); **run the big job on CUDA**.

**Track 4 — later.** Only after SFT translates: distill to 1B + GRPO/BLEU RL.

## Immediate plumbing check (while data builds)

Re-run the current 7.8 h with **LoRA r32 on MPS**, `--max-frames 280`, **8–10 epochs, eff-batch 8, resampled delay** — a cheap local capacity/soundness probe, not the real run. If it **memorizes** the seen set (loss → near-zero, seen rows translate) the pipeline is sound and it's purely data-starvation. If it **still loops on seen data**, there's a bug to fix before spending on data. (The real run goes full-model on CUDA — §Full-model, not LoRA — so a green LoRA probe is necessary-but-not-sufficient; it de-risks the data pipeline, not the full-finetune weights path.)

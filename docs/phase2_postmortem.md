# Phase-2 VI full-SFT post-mortem (run `vi_full_p2`, 2026-08-02/03)

Status at writing: 2-epoch main run **complete and below warm-start quality**; epoch-3
extension (flat 1e-5) running as the recovery experiment. Verdict section to be
finalized when it ends.

## Setup

| | Phase 1 (reference) | Phase 2 (this run) |
|---|---|---|
| Init | base 3B (`kyutai/hibiki-zero-3b-pytorch-bf16`) | warm start from phase-1 final (`model_step055284`, val128 chrF 19.6) |
| Data | 147k samples / 224 VI-h (PhoMT subset + FLEURS train) | 690k samples / 1,114 VI-h (full PhoMT cache + FLEURS train, `--max-frames 280`) |
| Box | H100 SXM 80 GB, torch 2.12.1+cu126 | H100 NVL 94 GB, torch 2.13.0+cu132, CUDA 13.2 driver 595 |
| Batch | 8 | 16 (+ causal-SDPA + torch.compile = 0.25 s/step, 1.4× phase-1 throughput) |
| LR | 1e-4 → 3e-5 @50%, warmup 500 | **5e-5 → 2e-5 @50% → 1e-5 @80%**, warmup 500 |
| Text weight | 5 → 2 @60% | 3 → 2 @50% |
| Steps | 55,284 (3 epochs) | 86,258 (2 epochs) + 43,129 extension (epoch 3, flat 1e-5, tw 2) |

## What happened (greedy val128, in-train, text-temp 0)

nonempty_chrf = chrF over only the non-empty predictions (rescored offline from
the saved `greedy_step*/` outputs).

| step | chrF | nonempty_chrf | nonempty/128 | phase |
|---|---|---|---|---|
| 0 (warm start) | ~19.6 (phase-1 standalone) | — | — | baseline |
| 9,000 | **1.24** | 8.22 | 17 | LR 5e-5 — **text-pad collapse** |
| 18,000 | 11.40 | 12.52 | 113 | partial self-recovery |
| 27,000 | 7.52 | 11.93 | 69 | oscillating, still hot |
| 36,000 | **13.04 (run best)** | 14.55 | 107 | backed up as `model_best_2ep` |
| 45,000 | 11.28 | 13.17 | 100 | after decay to 2e-5 |
| 54,000 | 11.84 | 15.48 | 85 | |
| 63,000 | 3.83 | 12.26 | 31 | epoch-2 re-collapse |
| 72,000 | 6.09 | **16.46** | 37 | |
| 81,000 | 6.40 | 14.72 | 43 | LR 1e-5 tail |
| 86,258 (final) | 7.48 | 13.67 | 55 | end of epoch 2 |

The nonempty_chrf column is the key read: through the epoch-2 re-collapse it
held at 12–16 (even peaking at 72k) while corpus chrF cratered. Translation
quality of what the model does say never degraded — the entire failure is the
pad/silence attractor going quiet on more and more inputs.

Teacher-forced val CE meanwhile: audio 2.83 → ~2.71 plateau from ~step 45k; text
2.61 → **bottom 2.39–2.40 @ step 40–48k → rose to 2.49 by the end** (train loss
still falling → overfit). TF CE *improved through the entire collapse* — it is
pad-dominated (57% of supervised text tokens are prefix pads; phase-1 plumbing
probe) and cannot see generation quality.

Infrastructure was NOT the cause: the identical eval path scored chrF 22.8 on the
warm start in the pre-launch smoke (step 10), and VRAM/throughput were nominal
throughout (93–94 GiB of 93.6, 0.25 s/step, zero crashes).

**Extension false start (fixed):** the first epoch-3 launch (`--lr-schedule
1e-5@0`) silently ran at **2e-5** — `optimizer.load_state_dict` on resume
restored the old run's param groups *including the custom `points` schedule*,
overriding the new flag (the old 5e-5/2e-5/1e-5 schedule re-stretched over
129,387 steps). Caught at step ~89.6k via the train log's `lr` field; fixed in
`train_lora.py` (resume now reasserts this run's schedule) and restarted from
`trainer_step086258`. Rule: after any resume, verify the logged `lr` matches
the flag.

## Diagnosis, in order of confidence

1. **Warm-start peak LR too hot (primary, config error).** 5e-5 > phase-1's final
   3e-5. A converged model perturbed at high LR falls into the nearest cheap
   attractor; the text stream's pad dominance makes "always emit pad" (= empty
   output, silent subtitles) a low-loss policy. From-scratch runs tolerate hot
   LRs precisely because there is no minimum to be kicked out of.
   **Rule: continuation training resumes at ≤ the prior run's final LR.**
2. **Pad-dominated text CE masks generation collapse.** Only dense greedy evals
   (`--eval-every 9000`) caught it — at step 9k, 40 min in, instead of at the end.
   Keep them; consider adding a content-only text CE (mask prefix pads) to the
   val logging so TF metrics can see this failure mode too.
3. **Epoch-2 synthetic-voice overfit.** Train = Kokoro TTS synthetic speech;
   val128 = real FLEURS audio. Val text CE rising after ~1 epoch over the full
   synthetic set while train loss falls is the same train/val divergence phase 1
   hit — a data-distribution ceiling, not a schedule problem. More passes over
   the same synthetic voices go negative after ~1 epoch at this scale.

## Recommendations

- **If epoch 3 (flat 1e-5) recovers past ~19.6:** low-LR continuation works;
  ship its best checkpoint and treat rule (1) as the only fix needed.
- **If it stalls (likely, low teens):** rerun **from scratch on the full 1,114 h
  with the proven phase-1 recipe** — lr 1e-4 → 3e-5 @50%, text-weight 5 → 2 @60%,
  warmup 500, 2 epochs, batch 16 + the speed config (0.25 s/step ⇒ ~6 h, ~$15).
  Phase 1 was explicitly data-limited at 224 h; 5× data with zero warm-start
  pathology is the cleanest high-expectation experiment.
  A conservative warm start (peak 2e-5) is the cheaper fallback but epoch 3 is
  already running that experiment in spirit.
- **Data beats schedule from here:** the epoch-2 val drift is the synthetic-only
  ceiling. Next data round should add real VI speech (CommonVoice, VIVOS) and
  more voice diversity before adding more Kokoro hours (see vi_training_plan).
- **Early-stop on val:** stop when val text CE turns up or greedy chrF flatlines
  (~1 epoch on full synthetic data); epochs past that point actively hurt.
- Keep best-on-chrF selection: it is the only reason the run's best state
  (chrF 13.0 @ 36k) survived the later collapse.

## Better val metrics (TF + nonempty_chrf IMPLEMENTED, live in the epoch-3 restart)

Text (teacher-forced, free — same forward as val CE; in `evaluate_teacher_forced`,
logged to `val_log.jsonl` and by `validate_lora.py`):
- **Content-only text CE** (`content_text_loss`) — CE over non-pad target tokens
  only. Plain text CE is 57% prefix pads and improved straight through the collapse.
- **Silence score** (`silence_score`) — at each target's first-content-token
  position, model probability mass on the pad token. Measures the pad/silence
  attractor directly; flags a collapse at the next val (2k steps) instead of the
  next greedy eval (9k).
- **Content token accuracy** (`content_acc`, top-1 on non-pad positions) —
  human-readable companion.

Text (greedy eval):
- **nonempty_chrf** now logged next to nonempty count in `greedy_eval_log.jsonl`
  so "collapsed" vs "bad translation" separate at a glance (validated offline —
  see table above).
- (still planned) **COMET (`wmt22-comet-da`)** on val128 outputs (needs
  `unbabel-comet` + vi source text from pairs; seconds on H100). Much better
  adequacy signal than chrF/BLEU; keep chrF for phase-1 comparability.

Audio (still planned — currently ZERO quality metric on the speech output, the
actual product):
- **ASR round-trip**: transcribe generated EN audio (faster-whisper on GPU),
  score WER/chrF vs reference — the paper's own eval style; catches audio-head
  regressions the text stream can't see. ~1–2 min per val128 pass.
- Keep the existing loop/silence gates (rms/peak, repeat4) as cheap sanity checks.

## Artifacts

- `finetune/runs/vi_full_p2/` on the box; checkpoints synced to HF model repo
  `huybik/hibiki-zero-vi-full-sft` via `hf_sync` (+`hf_transfer`).
- `model_best_2ep.safetensors` / `best_2ep.json` = 2-epoch best (chrF 13.04 @ 36k),
  protected from being overwritten by the extension's tracker.
- Full logs: `train_log.jsonl`, `val_log.jsonl`, `greedy_eval_log.jsonl`,
  per-step greedy predictions in `greedy_step*/`.
- Data: dataset repo `huybik/hibiki-zero-vi-full-sft` (Mimi cache chunks +
  `fleurs_cache.tar.zst`) is fully self-contained for reruns.

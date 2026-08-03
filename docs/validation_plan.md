# Vietnamese validation plan

This plan is the decision contract for the [data generation](data_generation_plan.md)
and [training](training_plan.md) plans. Its central rule comes directly from the
phase-2 post-mortem: teacher-forced validation can improve while free-running
generation collapses to silence, so only free-running outputs can qualify a
checkpoint.

## Goal and non-goals

The goal is to distinguish five failure modes: no Vietnamese grounding, silent
generation, bad translation when non-silent, bad English speech, and excessive
lag/over-generation. It must also catch regression on languages already
supported by the base model.

This suite does not tune on the test set, use synthetic speech as the primary
quality signal, collapse all failures into one BLEU number, or replace generated
outputs with teacher-forced predictions.

## Frozen evaluation sets

| Set | Role | Selection rule |
|---|---|---|
| `finetune/pairs/val16.jsonl` | Frequent collapse sentinel | Existing deterministic first 16 FLEURS validation rows. Fast alert only; never rank final checkpoints from 16 rows. |
| `finetune/pairs/val128.jsonl` | Checkpoint decision gate | Existing deterministic first 128 FLEURS validation rows. Keep fixed for historical comparison. |
| Full FLEURS validation | Milestone development result | Real VI speech only; evaluate less often than val128. |
| FLEURS test (347 pairs) | Final one-time report | Do not use for checkpoint selection, thresholds, or reward tuning. |
| New real-speech dev suite | Domain, speaker, accent, noise, and duration slices | Freeze speakers and normalized text hashes before training; no overlap with any train corpus. |
| Long-form/latency suite | Multi-sentence timing and silence recovery | 30–120 s real-source examples with sentence/word timing metadata. Required before T3. |
| Synthetic regression slice | Pipeline and memorization diagnostics | Report separately; it can never qualify a checkpoint. |
| Existing-language suite | Catastrophic-forgetting gate | Fixed FR/ES/PT/DE real-speech manifests, scored with the same runtime settings as their archived baseline. |

The new data builder must emit a train↔dev/test normalized-text overlap report.
Speaker-disjoint splits are required where corpus metadata permits. All ids and
ordering are immutable once the first training run begins.

## Evaluation ladder

### V0 — data and cache integrity

Run the D0–D4 gates from the data plan: waveform validity, transcript/translation
QA, split leakage, manifest/cache id agreement, and degenerate-code scan. A
model result is invalid if its data version cannot reproduce these artifacts.

### V1 — teacher-forced diagnostics

The existing command reports audio/text CE, content-only text CE, content
accuracy, and `silence_score`. On the H100 pod, activate `/venv/main` and use
`python`; locally substitute the mandated conda interpreter:

```bash
python finetune/validate_lora.py \
  --device cuda --dtype float32 \
  --cache-dir finetune/cache/validation \
  --adapter <checkpoint.safetensors> \
  --batch-size 8 --out-json <run>/validation_teacher_forced.json
```

This path deliberately keeps prefix-PAD weight 1.0 even if training used 0.5,
so its CE remains comparable across arms. The weighted training `text_loss`
does not. These metrics diagnose train/real divergence. They are explicitly ineligible
for checkpoint selection: the collapsed phase-2 model scored better than the
healthy phase-1 checkpoint on every teacher-forced metric, including content CE
and the proposed silence score.

### V2 — deterministic free-running text

This is the mandatory generation gate. Use text temperature 0, fixed ordering,
EOS stopping, and text-only output:

```bash
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --pairs finetune/pairs/val128.jsonl \
  --adapter <checkpoint.safetensors> \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --text-only \
  --out-dir <run>/eval_val128_greedy
```

Record corpus chrF/BLEU/WER, nonempty chrF, nonempty count, EOS count, mean
length ratio, overlong count, repeated-4gram count, and every prediction. Corpus
chrF answers "does the system translate?"; nonempty chrF separates translation
quality from silence collapse.

Use val16 every 3k steps as an alert and val128 every 9k steps or at each saved
milestone as the decision read. The current trainer accepts one in-process eval
set, so the second cadence requires a checkpoint watcher or trainer support;
running val128 every 3k is unnecessary cost.

Also run the production sampling configuration (`--text-temp 0.4 --audio-temp
0.8 --top-k 250 --top-k-text 250`) with a fixed seed at milestones. It measures
shipping behavior but does not replace deterministic V2 selection.

### Prefix-PAD weighting A/B

`--text-prefix-pad-weight` is a project experiment, not a paper setting. Weight
1.0 is the backward-compatible control; 0.5 reduces supervised prefix-PAD CE
mass while leaving content/EOS at 1.0, ignored tail/batch pads unchanged, and
audio loss unchanged. Static implementation checks have passed, but there is no
training evidence.

Compare T1A (1.0) and T1B (0.5) from the same base, data order, schedule, 9k-step
budget, and evaluation ids. Training must expose a seed before the formal A/B;
keep it identical. Report the
prefix-PAD/content/EOS counts for the exact cache variant alongside each run.

The 0.5 arm passes the pilot only if the final val128 read shows:

- corpus chrF and nonempty rate improve over the 1.0 arm at matched steps;
- nonempty chrF is no more than one absolute point below the control;
- EOS, loops, and mean length ratio still pass checkpoint eligibility;
- generated-audio ASR chrF is no more than one absolute point below the control,
  with no new silence, clipping, or non-finite failures;
- the improvement appears on the real-speech suite, not only synthetic rows.

The one-point tolerances are project proposals frozen before launch. A lower
weighted teacher-forced CE, lower pad probability under teacher forcing, or the
expected 45/55→29/71 loss-mass shift is not a win. Prefix weighting mitigates a
gradient imbalance; it does not test the free-running state where the model is
conditioned on its own pads.

### V3 — generated-audio round trip

Run `eval_lora.py` without `--text-only` to emit English audio:

```bash
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --pairs finetune/pairs/val128.jsonl \
  --adapter <checkpoint.safetensors> \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --out-dir <run>/eval_val128_audio
```

The current command writes WAV files but does **not** score their speech quality.
Required work: transcribe generated EN with a pinned ASR model and write ASR
WER/chrF/BLEU against the reference, plus per-row RMS, peak, clipping, duration,
silence ratio, and failure reason. The paper reports ASR-BLEU; it does not
provide our ASR model or thresholds, so those are frozen from phase-1 and base
audio baselines before comparing new checkpoints.

### V4 — streaming timing and long-form behavior

Required instrumentation must record, per generation frame:

- first non-pad text time and first non-silent audio time;
- text EOS and audio-tail time;
- prefix hypotheses at fixed source progress points;
- sentence-boundary pause and recovery behavior;
- source duration, output duration, and target reference timing.

From these, report first-content latency, end lag, output/input duration ratio,
prefix chrF, and a standard lag metric once the required word alignment exists.
The Hibiki-Zero reward scores every eight input words, but that interval is a
paper control, not evidence that one timing metric is sufficient for this
project. Current `eval_lora.py` retains only the final text and has no V4
instrumentation.

### V5 — robustness and forgetting

At milestone checkpoints, evaluate fixed real-speech slices for:

- clean versus noisy/reverberant source speech;
- short, medium, and long durations;
- speaker, gender, accent/region, and recording-corpus coverage where metadata
  permits responsible reporting;
- leading, internal, and trailing silence;
- FR/ES/PT/DE translation using archived baselines.

Report every slice even when small; do not hide a failed long/noisy slice inside
the aggregate. Old-language manifests and an automated comparison report are
required work.

## Checkpoint eligibility and stop rules

The following are project-proposed controls, frozen before T1 begins:

| Gate | Eligible checkpoint |
|---|---|
| Nonempty | At least 95% on val128 and no concentrated empty slice |
| EOS | At least 90% on val128 |
| Loops | Repeated-4gram failures at most 10% |
| Length | Mean prediction/reference word ratio at most 2.0 and no regression in overlong count |
| Adequacy | Corpus chrF is the primary selector among eligible checkpoints; nonempty chrF must not fall while corpus chrF rises |
| Audio | No non-finite/all-zero output; ASR and acoustic gates pass their pre-frozen baselines |
| Long-form | No systematic silence after an internal pause; timing metrics pass the T3 baseline |
| Forgetting | No more than 10% relative chrF regression on an existing-language aggregate |

The percentages are project proposals, not paper thresholds. Before the first
new run, score the archived phase-1 checkpoint and adjust a threshold only if it
would reject that known baseline for a documented measurement reason; then
freeze it for all comparisons.

Operational stop rules:

- A single low-nonempty read during early from-scratch grounding is a red alert,
  not an automatic kill. Stop if it fails to recover for two subsequent greedy
  reads or if nonempty chrF also degrades.
- Stop after real-val content CE turns upward and two free-running reads fail to
  improve. Falling train loss is not a reason to continue.
- Roll back immediately on non-finite loss, corrupt audio, checkpoint reload
  mismatch, or an LR that differs from `run_config.json`.
- Never overwrite the current best eligible checkpoint when a later epoch
  collapses.

## Stage decisions

| Transition | Go | No-go |
|---|---|---|
| T1 A/B → T2 mixed | 0.5 passes the matched pilot gates, or the result is inconclusive and the conservative 1.0 control is retained | Either arm has an implementation/audio regression, or the comparison was not seeded and matched. |
| T2 → T3 long-form | Real-speech aggregate and slices improve over T1 without forgetting | Gains exist only on synthetic rows or silence/loops worsen. |
| T3 → T4 free-running optimization | Translation is adequate, but a measured silence/latency defect remains | SFT does not translate; RL must not hide a grounding failure. |
| Candidate → release | Full V0–V5 report passes, including untouched test and existing languages | Any mandatory artifact or gate is missing. |

## Required implementation work

1. Extend best-checkpoint selection beyond chrF to the frozen eligibility gates.
2. Expose and log the training RNG seed so the prefix-PAD A/B is reproducible.
3. Add row-level audio/ASR scoring for WAVs produced by `eval_lora.py`.
4. Retain frame-level text/audio emission timing and compute V4 metrics.
5. Build immutable real-speech long-form and old-language manifests.
6. Add a comparison command that joins metrics by checkpoint and slice and emits
   one machine-readable decision report.
7. If COMET is added, pin its model/version and treat it as a secondary adequacy
   metric. It is not installed or implemented today.

## Reproducibility artifacts

Every evaluated checkpoint keeps:

- checkpoint and base-model SHA-256 hashes;
- repository commit, environment/package manifest, device, dtype, and generation
  parameters;
- data/cache version and pair-manifest hashes;
- `run_config.json`, `train_log.jsonl`, `val_log.jsonl`, and
  `greedy_eval_log.jsonl`;
- prediction CSVs, metrics JSON, generated WAVs for V3, ASR transcripts, and
  timing traces for V4;
- aggregate and per-slice comparison reports against phase 1, T1, and the base
  old-language model;
- random seed and ordered ids for every sampled evaluation;
- a final signed-off go/no-go record naming the selected checkpoint and failed
  alternatives.

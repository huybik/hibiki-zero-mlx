# Vietnamese validation plan

This is the decision contract for the
[base-start training plan](training_plan.md). Its central rule is unchanged:
teacher-forced validation can improve while free-running generation collapses
to silence, so only free-running outputs can qualify a checkpoint.

## Evaluation sets

| Set | Use | Selection status |
|---|---|---|
| `finetune/pairs/val128.jsonl` | Deterministic in-training health and historical comparison | Primary frequent gate; fixed 128-row FLEURS subset. |
| Full FLEURS validation (149 rows) | End-of-epoch VI→EN development result | Primary development domain. |
| VIVOS dev (1,106 rows, 5 held-out speakers) | Independent corpus/speaker/domain result | Required milestone domain after a flat eval manifest is built. |
| FLEURS test (347 rows) | Final report | Sealed until checkpoint selection is frozen. |
| VIVOS official test (760 rows) | Final independent-domain report | Sealed until checkpoint selection is frozen. |
| Synthetic slice | Pipeline and memorization diagnostics | Never qualifies a checkpoint. |
| FR/ES/PT/DE suite | Catastrophic-forgetting check | Required for release after manifests/reporting exist. |

The VIVOS development evaluator must use all 1,106 approved source rows and
their accepted Gemini English translations. Do not restrict text evaluation to
the 733 rows whose synthesized target audio passed training-cache QA; that
filter measures target TTS quality and would bias the speech-translation set.
Create `finetune/pairs/vivos_dev.jsonl` in the evaluator's existing flat schema:
`id`, `vi_audio`, and `text_en`, with immutable ordering and source/reference
hashes recorded beside the manifest.

## V0 — preflight baselines

Run the unadapted base model and the archived phase-1 checkpoint through the
same deterministic val128 command, on the same box and commit used for the new
run. The base result confirms the unsupported-language floor; phase 1 is the
healthy comparison baseline.

```bash
# Base model: intentionally omit --adapter.
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir finetune/runs/baselines/base_val128_greedy

# Repeat with the archived phase-1 full-model checkpoint.
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --adapter <phase1_model_step055284.safetensors> \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir finetune/runs/baselines/phase1_val128_greedy
```

The archived phase-1 reference is chrF 19.61, 128/128 nonempty, 126/128 EOS,
and 8/128 repeated-4gram failures. Recompute it in the new environment instead
of assuming the archived numbers are bit-identical.

## V1 — teacher-forced diagnostics

Training computes cached teacher-forced validation every 2k steps. Standalone:

```bash
python finetune/validate_lora.py \
  --device cuda --dtype float32 \
  --cache-dir finetune/cache/validation \
  --adapter <checkpoint.safetensors> \
  --batch-size 8 \
  --out-json <run>/validation_teacher_forced.json
```

Record total/audio/text CE, content-only text CE, content accuracy, and
`silence_score`. These diagnose optimization and train/dev divergence only.
They cannot select or qualify a checkpoint: the collapsed phase-2 model scored
better than the healthy phase-1 model on all of them.

## V2 — deterministic free-running selection

The trainer runs text-temperature-0 val128 every 9k steps. Its compact log is an
alert and raw-chrF tracker. For every candidate checkpoint, rerun the standalone
evaluator because its `metrics.json` also includes the full length and loop
metrics used by eligibility:

```bash
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --adapter <checkpoint.safetensors> \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir <run>/eval_val128_stepNNNNNN
```

Record corpus chrF/BLEU/WER, nonempty chrF, nonempty count, EOS count, mean
prediction/reference length ratio, overlong count, repeated-4gram failures, and
every prediction. Corpus chrF captures the combined translation-and-silence
result; nonempty chrF separates adequacy from silence collapse.

### Eligibility gates

Freeze these thresholds before training:

| Gate | Requirement on val128 |
|---|---|
| Nonempty | At least 122/128 (95%). |
| EOS | At least 116/128 (90%). |
| Loops | At most 12/128 repeated-4gram failures (10%). |
| Length | Mean prediction/reference word ratio at most 2.0; overlong count must not regress materially from the recomputed phase-1 baseline. |
| Adequacy | Rank eligible checkpoints by corpus chrF; nonempty chrF must not fall by more than one absolute point while corpus chrF rises. |

The percentages and one-point tolerance are project controls, not paper
thresholds. Adjust them only during V0 if the recomputed healthy phase-1
baseline fails for a documented measurement reason, then freeze them.

The full run uses prefix-PAD weight 0.5. It is acceptable only if these
free-running EOS, loop, length, nonempty, and adequacy gates pass. The expected
reduction in weighted training loss is not evidence of success; reject a model
that produces more text but becomes prematurely verbose, repetitive, or less
source-grounded.

`model_best.safetensors` is not the final selection because the trainer saves it
on raw chrF without eligibility checks. Select manually from saved 9k-step
checkpoints and preserve the paired trainer state.

## V3 — milestone real-speech validation

At every epoch boundary and before protecting a final candidate, run standalone
deterministic evaluation on:

1. full FLEURS validation; and
2. all 1,106 VIVOS dev rows.

Use the same command as V2 with the corresponding pair manifest and limit. Keep
per-corpus metrics separate and report a simple unweighted macro average only as
a summary; do not let the larger VIVOS set numerically erase a FLEURS regression.
A candidate must remain eligible on each domain. Among eligible candidates,
prefer the one with the stronger macro chrF, using val128 chrF for historical
tie-breaking.

The VIVOS manifest does not exist in the required flat schema today. Building
and hash-freezing it is required before final selection, but it does not block
the B1 training process from starting with val128 monitoring.

## V4 — production sampling and generated audio

For milestone candidates, repeat val128 with shipping text/audio sampling:

```bash
python finetune/eval_lora.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --adapter <checkpoint.safetensors> \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 \
  --text-temp 0.4 --audio-temp 0.8 \
  --top-k 250 --top-k-text 250 \
  --stop-on-eos --seed 42 \
  --out-dir <run>/eval_val128_production
```

The current evaluator writes WAVs but does not score their speech quality. Until
an audio scorer is implemented, require manual audition of a frozen sample and
basic file integrity, but do not claim ASR or acoustic gates. Release validation
still requires a pinned English ASR round trip plus per-row finite/RMS/peak,
clipping, duration, silence, and failure metrics.

## Stop rules

- One failed early 9k read is a red alert, not an immediate kill for a base-start
  model. Stop if the next two reads do not recover eligibility, or if nonempty
  chrF also declines.
- Stop after validation content CE rises and two consecutive free-running reads
  fail to improve.
- Stop immediately on non-finite loss, corrupt output, checkpoint reload
  mismatch, or logged LR/config disagreement.
- Do not overwrite an earlier eligible checkpoint when a later epoch collapses.
- Do not use test sets, synthetic rows, or teacher-forced metrics to rescue a
  failed development decision.

## Final test and release boundary

After selecting one checkpoint without looking at test data, run FLEURS test and
VIVOS official test exactly once. Report both separately. A research candidate
may be declared from V0–V4; a release candidate additionally needs the currently
missing generated-audio scorer and automated FR/ES/PT/DE forgetting suite.

Long-form latency and streaming-prefix metrics remain separate future work.
They are not claimed by the short-form B1 experiment.

## Reproducibility artifacts

For every evaluated candidate retain:

- checkpoint and base-model SHA-256 hashes;
- repository commit, environment manifest, device, dtype, and generation args;
- cache and pair-manifest hashes plus ordered evaluation ids;
- `run_config.json`, `train_log.jsonl`, `val_log.jsonl`, and
  `greedy_eval_log.jsonl`;
- prediction CSVs, metrics JSON, and generated WAVs where applicable;
- protected model/trainer pair and sync evidence;
- per-domain comparison against the recomputed phase-1 baseline; and
- the final go/no-go record, including failed and abandoned candidates.

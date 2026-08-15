# Vietnamese base-start training plan

This is the execution plan for the next VI→EN full training run. It replaces the
phase-2 continuation recipe with full-model SFT initialized
from the upstream Hibiki-Zero 3B base model. Checkpoint qualification is defined
in [validation_plan.md](validation_plan.md); the closed warm-start failure remains
documented in [phase2_postmortem.md](phase2_postmortem.md).

## Training objective

Train the strongest available VI→EN model from the base model on the full
1,114-hour PhoMT cache while preventing the free-running PAD/silence collapse
seen in phase 2.

The recipe makes two deliberate corrections:

- initialize from `weights/hibiki-pytorch-77f82164@110.safetensors`;
- do not pass `--init-adapter` or `--resume-checkpoint` at launch;
- use full-model SFT, not LoRA;
- keep the phase-1 schedule that first established Vietnamese grounding;
- set supervised prefix-PAD weight to 0.5 so content/EOS receive more of the
  text-loss mass;
- use deterministic free-running validation for all decisions.

"Base-start" means starting from the pretrained upstream base weights, not
randomly initialized weights. A trainer checkpoint may be used only to recover
an interrupted instance of this same run with the same data and configuration.
It must not be used to warm-start a new experiment.

## Why this run comes first

Phase 1 started from base on 224 VI hours and reached val128 chrF 19.61. Phase 2
warm-started that checkpoint on the 1,114-hour cache, used a hotter continuation
LR, and collapsed toward text PAD/silence even while every teacher-forced metric
improved. A verified low-LR continuation did not recover it.

The next run therefore restarts from base on the full cache with the phase-1
optimizer recipe. It also uses prefix-PAD weight 0.5 because this is a full
training run rather than an initialization A/B: PAD accounted for roughly 45%
of effective supervised text-loss mass at weight 1.0, while 0.5 shifts the
measured balance to roughly 29% PAD / 71% content+EOS. The implementation and
gradient boundaries are verified, but this remains its first scaled training
use. Do not additionally combine the run with a paper-optimizer rewrite,
long-form batching, or an unimplemented real/synthetic sampler.

## Frozen inputs

| Input | Role | Current state |
|---|---|---|
| `finetune/cache/phomt_stream` | Synthetic VI→EN training reservoir | 694,422 rows / 1,114 VI source hours; audited and published in eight cache chunks. |
| `finetune/cache/train` | Real FLEURS VI→EN training data | 1,449 rows / 4.44 VI source hours. |
| `finetune/cache/validation` | Teacher-forced FLEURS diagnostics | Existing fixed validation cache. |
| `finetune/pairs/val128.jsonl` | In-training free-running gate | Existing fixed 128-row FLEURS subset used by phase 1/2. |
| VIVOS dev source manifest | Independent real-speech milestone validation | 1,106 speaker-disjoint source rows with accepted Gemini EN references; the flat eval manifest still must be built. |
| VIVOS Mimi cache v2 train | Future source-aware training experiment | 7,024 rows / 10.00 source hours; local release is complete, but Hub publication is not verified. |

The first run intentionally excludes the VIVOS training cache. Uniform pooling
would give FLEURS+VIVOS only about 1% of row exposure, while the current trainer
cannot enforce or log a source-aware mixture. Adding VIVOS under that boundary
would not implement a meaningful real-data policy. Build the VIVOS dev eval
manifest before final checkpoint selection;
its source rows are valid independently of target-audio QA acceptance.

## Stage B0 — preflight

Before renting the training box:

1. Verify the base-model, PhoMT cache, FLEURS cache, and pair-manifest hashes.
2. Restore the exact base weight named above. Do not restore a finetuned model
   into the training run directory.
3. Run deterministic val128 generation for the unadapted base model and the
   archived phase-1 checkpoint in the new environment.
4. Run a 10-step full-model smoke from base. Verify finite loss, the requested
   LR, checkpoint reload, greedy artifacts, and archive sync.
5. Record the sample count and optimizer steps after `--max-frames 280`; do not
   infer them from the unfiltered cache total.

The smoke is valid only if `run_config.json` contains no `init_adapter` or
`resume_checkpoint`, and the saved model reloads against the same base hash.

## Stage B1 — full-model SFT from base

Use fp32 master weights with CUDA bf16 autocast, batch 16, warmup 500, LR
`1e-4→3e-5` at 50% of the run, text weight `5→2` at 60%, prefix-PAD weight 0.5,
gradient clip 1.0, and the trainer's current AdamW defaults. This is the
project-proven optimizer recipe plus the targeted PAD-loss correction, not a
paper-faithful optimizer reproduction.

The hard budget is two epochs. The expected selected checkpoint may occur in
the first epoch; the second epoch continues only while the validation stop rules
remain healthy.

```bash
python finetune/train_lora.py \
  --device cuda --dtype float32 --full-finetune \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --cache-dir finetune/cache/phomt_stream finetune/cache/train \
  --val-cache-dir finetune/cache/validation \
  --batch-size 16 --max-frames 280 --sort-by-length \
  --epochs 2 \
  --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
  --text-weight-schedule "5@0,2@0.6" \
  --text-prefix-pad-weight 0.5 \
  --seed 42 \
  --val-every 2000 --val-batch-size 8 \
  --eval-every 9000 \
  --eval-pairs finetune/pairs/val128.jsonl \
  --eval-limit 128 --eval-batch-size 8 --eval-text-temp 0 \
  --save-every 3000 --keep-checkpoints 3 --log-every 10 \
  --out-dir finetune/runs/vi_base_full_v1
```

The command is supported by the current trainer and starts from base because it
omits both checkpoint-initialization flags. `--dtype float32` is required for
fp32 master weights; CUDA forward passes use bf16 autocast automatically.

Start `finetune/hf_sync.py` alongside the run. At every 9k greedy read, protect
the matching model/trainer pair before checkpoint rotation if it becomes the
best eligible checkpoint. The trainer's `model_best.safetensors` tracks raw
chrF only and does not preserve a paired optimizer state.

## Monitoring and stop rules

Teacher-forced validation every 2k steps is diagnostic only. Free-running
val128 every 9k steps decides whether training remains healthy.

- A single poor early free-running read is a warning, not an automatic stop.
- Stop if nonempty output fails the eligibility gate for two additional 9k
  reads, or if nonempty chrF degrades at the same time.
- Stop if validation content CE turns upward and two consecutive free-running
  reads fail to improve.
- Stop immediately on non-finite loss, corrupt artifacts, checkpoint reload
  mismatch, or a logged LR different from `run_config.json`.
- Do not continue into another epoch merely because training loss falls.
- Never overwrite or delete the best eligible checkpoint when a later read
  collapses.

After each promising 9k checkpoint, run the standalone deterministic evaluator
to obtain the full eligibility metrics omitted from the in-training summary.
At epoch boundaries, also run the production sampling configuration and the
independent VIVOS dev suite described in the validation plan.

## Decision

The run is a **go** only if an eligible checkpoint:

1. beats or credibly matches the archived phase-1 val128 chrF 19.61;
2. preserves nonempty, EOS, loop, and length health;
3. improves or holds on the independent VIVOS dev domain; and
4. does not show the phase-2 pattern of falling train loss with worsening
   free-running real-speech output.

If no checkpoint passes, the base-start plus PAD-corrected recipe did not solve
the scaling failure. Do not rescue the run with a checkpoint continuation, a
mid-run prefix-weight change, or a third epoch. The next experiment must address
the data/sampling boundary.

## Deferred follow-ups

These are separate experiments and must not be folded into B1:

- **Real-source training mix:** add source-aware sampling and exposure logging
  at the dataset/sampler boundary, then train a new matched base-start run with
  VIVOS train. Repeating `--cache-dir` paths or silently duplicating rows is not
  an acceptable mixture implementation.
- **Longer sequences:** benchmark the implemented frame-budget schedule
  `288:10,384:8,512:5` with `--max-frames 512` and at least 5 GB VRAM headroom.
  It retains nearly all existing PhoMT+FLEURS rows, but it changes batch and
  sample exposure and is not part of this full-run configuration.
- **Paper optimizer:** expose AdamW betas, weight decay, and cosine decay before
  testing the paper's optimizer. The current trainer uses PyTorch defaults and
  a piecewise-constant schedule.
- **Long-form continuation and GRPO:** remain conditional on a healthy SFT model
  and the missing alignment/rollout infrastructure.

## Required artifacts

Preserve the base/data hashes, repository commit, environment manifest,
`run_config.json`, training/validation logs, every greedy prediction directory,
all standalone metrics JSON files, protected candidate model/trainer pairs, sync
logs, failures, corrective actions, and the final go/no-go decision. A result
without the exact initialization and data evidence is not part of the research
record.

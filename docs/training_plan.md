# Vietnamese training plan

This is the execution plan for the next VI→EN model. Data requirements live in
the [data generation plan](data_generation_plan.md); model-selection and release
gates live in the [validation plan](validation_plan.md). The phase-2 failure is
documented in [phase2_postmortem.md](phase2_postmortem.md).

## Goal and non-goals

The goal is a full-model supervised checkpoint that grounds Vietnamese speech,
remains non-silent in free-running generation, handles longer timing, and is a
safe input to a later free-running optimization stage.

This plan does not use LoRA for the shipping language adaptation, does not judge
checkpoints from teacher-forced CE, does not warm-start the coarse run from the
collapsed phase-2 trajectory, and does not begin RL before SFT translates.

## Evidence ledger

Settings below have three different authorities and must not be blended:

| Authority | Established result |
|---|---|
| **[Original Hibiki paper](https://arxiv.org/abs/2502.03382v2)** | Coarse ST: LR 3e-5, batch 96, 150k steps. Long-form supervised fine-tune: LR 2e-6, batch 8, 8k steps on about 900 h. AdamW weight decay 0.1, betas (0.9, 0.95), cosine scheduling. Later-stage warmup/minimum LR and loss weights are not reported. |
| **[Hibiki-Zero paper](https://arxiv.org/abs/2602.11072v1)** | New-language supervised fine-tune: LR 1e-6, batch 16, about 1k steps; then distillation and GRPO. GRPO reports LR 2e-7, batch 32, group 4, 2k steps. These are paper settings, not validated VI settings. |
| **Project observation** | Full SFT on 224 VI h reached val128 chrF 19.61; LoRA learned target-side continuation but not source routing. The successful run used LR `1e-4→3e-5`, text weight `5→2`, warmup 500, batch 8, and the trainer's current AdamW defaults. |
| **Project observation** | Warm-starting at 5e-5 caused text-pad collapse. A verified 1e-5 continuation did not recover it. Teacher-forced metrics scored the collapsed checkpoint better; only free-running eval detected the failure. |
| **Project observation** | Phase 1 ended with 0/128 empty, 126/128 EOS, 8/128 repeated-4gram loops, chrF 19.61, and about 1.9× mean over-generation. Synthetic-only validation drift appeared after roughly one effective pass. |
| **Implemented project experiment, not yet trained** | `--text-prefix-pad-weight` now weights supervised prefix PAD inside text CE while content/EOS remain weight 1.0, tail/batch pads remain ignored, and audio loss is unchanged. Default 1.0 is backward-compatible; 0.5 changes one measured effective PAD/content+EOS balance from about 45/55 to 29/71. Syntax, masking, gradients, audio invariance, and backward compatibility were checked, but no training result exists. Neither paper reports this technique. |

Important implementation fact: `train_lora.py` currently constructs AdamW
without explicit beta or weight-decay arguments, so PyTorch defaults apply
(betas `(0.9, 0.999)`, weight decay `0.01`). The paper's optimizer values are
**not** currently exposed as flags. Any paper-optimizer experiment requires code
work and must not be described as the same run as the project-proven control.
The existing `value@fraction` LR schedule is piecewise constant, not cosine;
cosine scheduling is another required feature for a paper-derived optimizer run.
`finetune/phase2.sh` currently opts into prefix-PAD weight 0.5, but no training
run has validated that edit; the trainer default and T1A control remain 1.0.

## Stage T0 — preflight

Before renting the box:

- For T1, verify the published PhoMT/FLEURS cache hashes, zero-degenerate scan,
  and pair manifests. For T2 and later, pass all D0–D4 gates and freeze the new
  cache/manifest hashes.
- Restore base weights, FLEURS train/validation caches, val128 pairs, and the
  real-speech development manifests.
- Run base and phase-1 checkpoints through the validation plan to establish the
  exact environment baseline.
- Run a short full-model smoke and verify finite loss, expected LR, checkpoint
  reload, greedy generation, and archive sync.
- Calculate steps from accepted samples after `--max-frames`; never copy a paper
  step count without comparing audio and sample exposure.

**Gate:** a checkpoint saved by the smoke reloads identically, the logged LR
matches the requested schedule, greedy artifacts are complete, and the baseline
metrics match their archived values within deterministic variation.

## Stage T1 — short prefix-PAD A/B pilot

Do not fund two full synthetic-only campaigns to validate one loss coefficient.
T1 is a short, matched pilot on the existing PhoMT+FLEURS cache. Both arms start
from the same base 3B checkpoint; T1A keeps the backward-compatible weight 1.0
and T1B changes only that value to 0.5. Its question is narrow: does weaker
prefix-PAD pressure improve early free-running behavior without hurting content
or audio? It cannot establish final quality, add long-form alignment, or replace
T4 on-policy optimization.

Expose `--seed` before calling the result definitive. On the H100 pod, after
activating `/venv/main`, run each arm for 9k optimizer steps with greedy val16 at
3k/6k/9k, then run val128 once on both final checkpoints. The short run uses a
static hot phase because compressing the historical fraction schedule into 9k
steps would decay it prematurely:

```bash
# BLOCKED until train_lora.py exposes --seed.
python finetune/train_lora.py \
  --device cuda --dtype float32 --full-finetune \
  --cache-dir finetune/cache/phomt_stream finetune/cache/train \
  --val-cache-dir finetune/cache/validation \
  --batch-size 16 --max-frames 280 --max-steps 9000 \
  --lr 1e-4 --warmup-steps 500 --text-loss-weight 5 \
  --text-prefix-pad-weight 1.0 --seed 42 \
  --val-every 2000 --eval-every 3000 \
  --eval-pairs finetune/pairs/val16.jsonl --eval-limit 16 \
  --save-every 3000 --keep-checkpoints 3 --log-every 10 \
  --out-dir finetune/runs/pad_ab_w1
```

T1B repeats that command from base with only:

```bash
--text-prefix-pad-weight 0.5 \
--out-dir finetune/runs/pad_ab_w05
```

`--seed` is required work and the displayed command is blocked until that flag
exists. `float32` means fp32 master weights with CUDA bf16 autocast. Do not reuse
an optimizer checkpoint between arms. `phase2.sh` is a historical launcher for
the failed 5e-5 warm start; adding weight 0.5 there did not make that recipe safe
and it is not the T1 command.

**Gate:** choose 0.5 only if it wins the matched free-running and generated-audio
checks in `validation_plan.md`. If the arms are indistinguishable or noisy, keep
1.0. Lower weighted teacher-forced loss and the expected loss-mass shift do not
count. Continue neither pilot into the shipping run.

## Stage T2 — mixed real/synthetic coarse SFT from base

Train from the base checkpoint on the D4 data mixture; do not warm-start from T1
or the collapsed phase-2 trajectory. Use the T1-winning prefix weight, or 1.0 if
T1 was inconclusive.

For the first mixed-data run, preserve the project-proven from-base optimizer
recipe: fp32 masters/bf16 autocast, batch 16, warmup 500, piecewise LR
`1e-4@0,3e-5@0.5`, text weight `5@0,2@0.6`, gradient clip 1.0, and the current
AdamW defaults. This is the only recipe that has routed VI successfully. It is a
project control, not paper-faithful, and every optimizer value must be logged
explicitly before launch.

The launch is blocked on source-aware sampling. Multiple `--cache-dir` values
currently pool rows uniformly, so no real/synthetic exposure ratio can be
guaranteed. Implement the mixture policy in the dataset/sampler boundary and log
sample counts by stratum at every evaluation interval. A hypothetical mixture
flag is intentionally not shown here because it does not exist.

Proposed exposure policy:

- at least 20% real-source examples per optimizer window;
- the remaining examples from the accepted synthetic supplement and FLEURS;
- at most two passes over the real-source stratum; rotate/subsample the synthetic
  reservoir instead of exhausting it at the cost of repeatedly replaying real rows;
- versioned source-noise cache variants from D4; the current cached-code trainer
  cannot augment waveforms online;
- early stop controlled by real free-running validation.

The 20% floor is a project proposal. Run one ablation only after the baseline:
change the mixture, not LR, optimizer, text weight, or initialization at the
same time.

The original-paper optimizer is a separate optional T2P experiment: peak LR
3e-5, AdamW betas `(0.9, 0.95)`, weight decay 0.1, cosine decay. Run it only after
those controls exist in code, and never label a current piecewise/default-AdamW
run as equivalent. Keeping it separate prevents a simultaneous data+optimizer
change from obscuring the first mixed-data result.

**Gate:** T2 must beat the archived phase-1 checkpoint on the frozen real-speech
suite while preserving free-running health. If it only improves synthetic
slices, the new data recipe has not solved the domain gap.

## Stage T3 — cold long-form supervised continuation

Initialize from the best eligible T2 checkpoint and train only on the aligned
long-form/real continuation slice. This is where the original Hibiki paper's
fine-tuning recipe is relevant.

Use the reported peak LR **2e-6** as the paper-derived reference, full-model
training, and an audio-exposure budget calculated from the final long-form
manifest. The first implementation uses a project-defined cosine from 2e-6 to
zero over one real-data pass, with no claimed warmup; exact reproduction is
impossible because the paper does not report the later-stage warmup or minimum.
The paper's
8k steps × batch 8 is reported context, not a target to copy: our sequence-length
distribution and global-batch interpretation differ. No warmup value is claimed
because the paper does not report one for this stage.

Keep the T1-winning prefix-PAD weight as a project control only if it remained
healthy on long-form timing slices. Long-form alignment changes the proportion
and meaning of prefix PAD, so recalculate the PAD/content/EOS balance and rerun a
small 1.0 versus 0.5 gate before a full continuation. The paper-derived LR does
not make prefix weighting paper-derived.

Before launch, benchmark the longest length buckets to select batch size and
gradient accumulation. The existing `--max-frames 280` would delete examples
longer than about 22 seconds and therefore cannot be used unchanged for this
stage. The exact command is frozen only after that memory benchmark and the
accepted sample count are recorded.

**Gate:** long-form and latency slices improve without more than a one-point
absolute chrF regression on the core real-speech set, and all free-running
eligibility gates continue to pass. Do not continue merely because
teacher-forced loss falls.

## Stage T4 — free-running optimization, conditional

Run this stage only if T2/T3 translates but still shows silence, excessive lag,
or unstable speaking behavior. Hibiki-Zero uses GRPO over the model's own
rollouts with a process/final BLEU reward; empty output naturally receives near
zero reward. Reported controls include `α=0.4`, scoring every 8 input words,
temperature 0.8, top-k 250, group 4, batch 32, LR 2e-7, and 2k steps.

There is no GRPO trainer, rollout store, prefix scorer, or reward implementation
in this repository today. T4 is required work, not an available command. For VI,
chrF may be used as a proposed reward only after it is A/B-checked against BLEU
and human adequacy on the frozen development set. Preserve an SFT KL/reference
control and select on held-out free-running validation, never reward on val128.

**Gate:** T4 must improve nonempty/latency behavior and corpus adequacy without
increasing loops, over-generation, or old-language regression. Otherwise ship
the best eligible SFT checkpoint.

## Validation and checkpoint policy

- Teacher-forced metrics are diagnostics for overfit and adequacy only.
- Use val16 as a frequent collapse sentinel and val128 as the decision set:
  val16 every 3k steps during T1, val128 at 9k and each later milestone. The
  current trainer supports only one in-process set, so T2 requires either a
  second evaluator process or multi-tier evaluation support before launch.
- Keep teacher-forced validation unweighted across PAD-loss arms. The current
  validation path intentionally uses prefix weight 1.0, which preserves metric
  comparability even when the training objective uses 0.5.
- A checkpoint is eligible only if it passes the free-running gates in
  `validation_plan.md`. Among eligible checkpoints, select highest corpus chrF,
  then use real-speech/audio metrics as tie-breakers.
- `model_best.safetensors` alone is not sufficient evidence. Preserve its
  paired trainer checkpoint, `best.json`, logs, predictions, run config, data
  hashes, and environment manifest.
- After any resume, inspect the first logged LR. A fixed bug previously allowed
  optimizer state to override the requested continuation schedule.

## Required code work before T2+

| Boundary | Required change |
|---|---|
| Data loader/sampler | Preserve cache stratum and enforce/log a source-aware mixture. |
| Trainer optimizer | Expose AdamW betas, weight decay, and cosine scheduling before any paper-optimizer experiment. |
| Reproducible A/B | Expose/log a training seed before declaring a prefix-PAD-weight winner. |
| PAD accounting | Log raw prefix-PAD/content/EOS counts and weighted mass; `text_loss` alone is not comparable between 1.0 and 0.5. |
| Trainer selection | Make eligibility thresholds and real-speech suite metrics part of best-checkpoint selection; current code selects on chrF only. |
| Evaluation cadence | Support frequent sentinel plus milestone val128 without blocking the training loop on val128 every 3k steps. |
| Alignment augmentation | Consume the versioned per-sentence/long-form alignment representation rather than a single baked clip delay. |
| Long-form training | Benchmark and configure variable length without the 280-frame truncation used by short-form SFT. |
| RL | Implement rollout generation, prefix/final reward, group normalization, reference/KL control, and resumable artifacts. |

## Go/no-go summary

- **Go T1A/T1B:** seed logging is implemented; 1.0 remains the control and 0.5
  is adopted only after a matched free-running/audio win.
- **Go T2:** D4 data passes and mixture sampling is implemented.
- **Go T3:** T2 passes real-speech free-running gates.
- **Go T4:** supervised translation is adequate but a measured free-running
  silence/latency failure remains.
- **No-go:** LoRA shipping run, hot warm start, checkpoint selection by CE, a
  third synthetic-only epoch after real-val divergence, or changing data,
  optimizer, and schedule in the same comparison.

# GRPO

Read when touching `training/finetune/grpo.py`, `grpo_profile.py`, `rewards.py` or
`h100.sh grpo`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Checks](#checks) · [Running](#running)

The stage after SFT ([plan](../grpo_plan.md)): the paper's GRPO with BLEU process rewards, on
streams of the `--cache-dir` caches. The first run is on the two-way model, both directions
(user, 2026-11-10; [plan](../vi_en_parity_plan.md#4-grpo) lists what that adds: both
directions' evaluation sets, a promotion rule over both, the Vietnamese tokenizer for the
EN→VI rewards). Checkpoints, resume and logs are the SFT run's
([sft-recovery](sft-recovery.md)), and evaluation is SFT validation's free-running pass
([sft-evaluation](sft-evaluation.md#checkpoint-selection)).

## Design

- **A window** is `--tau` (20) updates. Each window:
  1. a bf16 copy of the policy samples `--group` (4) rollouts of `--frames` (T, 1,200) frames for
     each of the window's `--streams` (32) × 20 streams. Updates take the planned streams in a
     seeded order: planning keeps a document's rows together, so in plan order one long
     recording (~75 streams) would fill 2–3 updates;
  2. CPU workers compute the rewards as each generation batch lands;
  3. the policy scores every rollout, giving π_old's logprobs.
- **After a rollout's first text EOS**, its text tokens and its audio from the EOS frame on stay
  out of the loss (`before_eos`), as SFT streams end. No reward reads them, so training them would
  only add noise; in a single-row stream they are most of the frames. They stay in place as
  inputs: set to -1, they would change the context of the delayed codebooks in the two frames
  before the EOS (shown on a tiny CPU model). `eos_rate` in the train log is the share of
  rollouts that ended.
- **An update** takes its own 128 rollouts in `--micro-batch` forwards, cut within its own rows.
  π_old's pass cuts them the same way, because another batch shape picks other kernels. Every
  rollout is used by exactly one update.
- **Rollout layout:** rollouts replay the stream's cached Vietnamese through `LMGen` with
  `stop_on_eos` off, and are stored in the cache's training layout (`grpo.rollout`).
  - `LMGen` reads the input given at step s as source frame s − 1 (step 0's input is never read).
    It samples text frame s at step s, and audio codebook k's frame s − delay_k; the delays are
    0 for text and codebook 0, 2 for the rest.
  - The text and audio hooks capture every sampled token, including those `step()` never returns.
    The hooks copy each token, because CUDA graphs reuse their output buffers.
- **π_old's logprobs** come from the update's own forward: training mode, gradients enabled, the
  result detached and never backpropagated. A no-grad forward picks other bf16 kernels, so the
  first update's ratio would not be 1.
  - Logprobs are taken at the sampling temperature 0.8; top-k is ignored. They are SFT's loss
    kernel negated (`common.token_ce`, dense over the full logits), so the two trainers share
    one kernel ([sft-training](sft-training.md#objective)).
  - Text is weighted `--text-weight` 100 against 1 for each audio codebook. Masked positions are
    the training forward's own.
- **Rewards** (`rewards.py`, the paper's Eq. 3–6):
  - Rewards sit at frame boundaries. A reward at b scores the text of the frames before b, and
    R_t sums the boundaries after t, so a reward credits exactly the tokens it counted.
  - y_b joins the references of the rows whose Vietnamese started before b, the row being spoken
    included. A row's reference is its cache text stream.
  - Texts are decoded as evaluation decodes (`common.decode_text_batch`): tokens above PAD, and a
    rollout's text ends at its first EOS, so an early EOS loses the rest of the translation.
  - A boundary follows every 31 frames (2.5 s) of Vietnamese speech, plus one at T, because the
    cache has no Vietnamese words.
  - Normalization uses the group's unbiased std plus 1e-4, so a frame where every rollout scores
    the same gets 0.
- **Evaluation** runs at update 0 and every `--eval-every` (200) updates, SFT validation's
  free-running pass on `EVAL_PAIRS`' five sets (FLEURS dev, the clean and noisy en-vi-val cv27
  and libriheavy). The first set of each direction selects (`common.selecting_sets`; FLEURS dev
  alone with these sets): promotion goes to the best mean chrF over them among checkpoints that
  pass the gates and whose median end offset on each is at most `LATENCY_SLACK_S` (0.2 s) past
  update 0's.
  - The slack: in the dry run, update 2 (two updates at 2e-7, in effect the start) had the
    better chrF (39.90 against 39.29) but measured 4.6 s against 4.5 s. Without slack it was
    blocked, and a blocked checkpoint is not kept (2 pairs).
  - `val_log.jsonl` records each evaluation's `healthy`, `on_time` and `promoted` (this step is
    now the best).
  - That baseline is stored as `start_end_offset_s` (by selecting set) in `run_config.json`, which
    `hf_sync` uploads when its size changes (adding the key at update 0 does). The logs reach the
    Hub only at the end, so a restore needs this file.
- **Length:** `--updates` defaults to the paper's 2,000. Every evaluation promotes, so a run that
  has stopped improving is simply stopped; its stop cannot be extended.
- **Losing work:** `hf_sync` uploads only evaluation-step pairs, so a lost pod costs up to
  `--eval-every` (200) updates. A preemption on the same disk costs at most a window.
- **Saves** land on window boundaries (`--save-every` 20 updates, and `--updates` a multiple of
  τ). A resume starts its window again with the same seeds, unique to each window and batch
  (32-bit, from `SeedSequence((seed, step, batch))`).
- **Reward workers** are forked when the pool starts, before CUDA and any thread. A fork pool
  starts all its workers at the first submit only from Python 3.11, which GRPO requires.
  - Forked workers also hold the launcher's run lock, and a pool never notices its parent dying.
  - So each worker sets `PR_SET_PDEATHSIG` and dies with the trainer. Otherwise a killed trainer
    would leave workers holding the lock, and every relaunch would exit "already running".

## Checks

On the Mac (MPS, bf16 upstream model, 2026-11-04):

- **Reward math:** on a hand-built stream, an on-time correct rollout beats the same words late,
  which beats one wrong word, which beats silence. Advantages are suffix sums of the normalized
  rewards.
- **Rollout alignment:**

  | Text logits, `LMGen` against the training forward | Same frame | ±1 frame |
  |---|---:|---:|
  | Argmax agreement | 100% | 55% |
  | Max difference | 0.20 | 29 |

  99.87% of sampled audio tokens sit inside the teacher-forced top 250; the rest are bf16
  near-ties.
- **The loss** against its own grad-enabled old logprobs:
  - ratio 1.000000, nothing clipped, and the loss equals −Σ c·A;
  - a no-grad old pass gave a ratio of 1.0012 and clipped 6 tokens;
  - one step along the gradient raised Σ c·A·logp.

The CUDA profile and dry run are under [Running](#running).

## Running

- **Optimize it heavily before its full run, on the box it runs on** (user, 2026-11-10: "remember
  to heavy optimize grpo when it phase start after sft"). `grpo_launch.sh`'s sizes (generation
  batch 512, micro-batch 4) are the H200's and do not fit an 80 GB H100 (512 peaked at 107 GiB;
  the H100's 2026-11-06 profile fit 192 and 2). Levers measured below but not built: rollouts cut
  to their batch's longest stream plus a tail and micro-batches cut to their last kept frame
  (about a third of generation and update; branch `grpo/stream-length`, WIP, never run on a GPU;
  it moves where the last reward sits, so ask the user), the GPU idle between stages (71% busy
  on the H200), generation bound by its step count (~13 ms a step at any batch). Compare each on
  the same streams by updates per GPU-hour.
- **Profile first, on the pod**, with `nvidia-smi` sampled beside it:
  `python -m training.finetune.grpo_profile --cache-dir … --model-weight START --out-dir …`.
  - It times generation at each `--gen-batch`, the rewards serially and in workers, and the
    old-logprob pass and the update at each `--micro-batch`.
  - It holds the fp32 policy and AdamW's 8-bit states resident throughout (`common.adamw`).
  - It times each generation size on a second call, after compilation. Each batch's CUDA graph
    capture stays in the time, as in a run.
  - An out-of-memory size is recorded and skipped, and `profile.json` is rewritten after each
    stage.
- **Profiled on the box** (2026-11-06, 700 W H100 SXM, from `tranche_epoch1b`'s final model, the
  tranche caches, T 1,200, fp32 policy and fp32 AdamW states resident: 18.7 GB more than now):

  | Stage | Measured |
  |---|---|
  | Generation, batch 64 / 128 / 192 | 17.5 / 22.2 / 27.0 ms a step; 3.05 / 4.80 / 5.91 rollouts/s; peak 49 / 57 / 66 GiB |
  | Rewards, 48 groups | 0.65 s serial, 0.14 s in 8 workers |
  | π_old's logprobs, micro-batch 2 | 43,900 frames/s |
  | Update, micro-batch 2 | 14,400 frames/s, peak 74 GiB; 3 and 4 run out of memory |
  | First update's ratio | exactly 1 (max deviation 0) |

  So the defaults are `--gen-batch` 192 and `--micro-batch` 2: an update of 128 rollouts takes
  ~22 s of generation, ~3.5 s of old logprobs and ~10.6 s of update, ~36 s (the plan estimated
  ~50 s); 1,000 updates ~10 h plus ~30 min of evaluations. The rollouts' mean final BLEU from
  that model was 26.2 (on the first streams in plan order, before the seeded order). The
  profile's GPU sat idle 39% of its 5.3 min, between its stages.
- **Profiled on the H200 box, alone on the GPU** (2026-11-10, 700 W H200 SXM, from round 4's
  final model, the tranche caches, T 1,200; the 2026-11-09 profile there ran beside the cache
  build and counted the leaked generator, so its micro-batch 4 "ran out of memory"):

  | Stage | Measured |
  |---|---|
  | Generation, batch 192 / 384 / 512 / 640 | 25.1 / 36.6 / 44.4 / 52.3 ms a step; 6.36 / 8.74 / 9.60 / 10.17 rollouts/s; peak 66 / 91 / 107 / 124 GiB |
  | Rewards, 160 groups | 2.9 s serial, 0.43 s in 8 workers |
  | π_old's logprobs, micro-batch 2 / 3 / 4 / 6 | 53,100 / 57,900 / 58,900 / 62,300 frames/s |
  | Update, micro-batch 2 / 3 / 4 / 6 | 16,200 / 18,600 / 19,300 / 20,400 frames/s; peak 68 / 96 / 111 / 134 GiB; 8 runs out of memory |
  | First update's ratio | exactly 1 at every size |

  So on the H200 `--gen-batch 512 --micro-batch 4` (`grpo_launch.sh`): an update of 128
  rollouts takes ~13.3 s of generation, ~2.6 s of old logprobs and ~8.0 s of update, ~24 s
  against ~31 s at 192 and 3; 800 updates ~5.3 h plus the evaluations. 640 and 6 gain 6% each
  and leave 17 and 7 GiB of the card. A generation step costs ~13 ms whatever the batch, so
  generation, 55% of an update, is bound by steps: the streams average 599 frames (longest 929)
  and every rollout runs 1,202 steps, and every update forward spans 1,200 frames though the
  loss keeps none after a rollout's EOS. Rollouts cut to their batch's longest stream plus a
  tail, and micro-batches cut to their last kept frame, would save about a third of both; not
  built (it changes where the last reward sits; the user's call). The profile's GPU read 71%
  busy on average, under 50% in 29% of its samples (between stages, and generation's short
  kernels).
- **The start policy** loads as SFT's does (`common.load_policy`), so its text EOS input row
  is reset to PAD's ([sft-training](sft-training.md#trainer)); until 2026-11-10 GRPO kept the
  checkpoint's own row. Rollouts and scoring share the row, and the first ratio is still 1.
- **A failed evaluation promotes nothing**, though its pair still reaches the Hub. The
  non-empty gate allows a tenth of a set's rows since 2026-11-10: at 8 of 180 a two-update dry
  run failed both its evaluations on round 4's silent Common Voice rows alone
  ([sft-evaluation](sft-evaluation.md#checkpoint-selection)).
- **Dry run** (2026-11-06, `h100.sh grpo grpo_dryrun`, τ 2, 4 updates of 4 streams × 4, stopped
  before its checkpoint sync when GRPO was deferred): every update's ratio mean 1.000–1.001, the
  loss finite (−925 to 368), but the gradient norm before clipping 0.42–0.95 million against SFT's
  ~8–15. The resume and the Hub's pair were not reached.
- **The gradient norm's scale** (checked 2026-11-07): the loss sums the weighted objective over
  every kept token of the update's 128 rollouts (no mean), so its norm runs ~10⁵ times SFT's
  (a mean loss). Grad clip 1.0 rescales it to 1 every update, as it already does SFT's ~10, and
  AdamW's step does not depend on the gradient's scale: after the clip a parameter's gradient is
  ~1/√3.1e9 ≈ 2e-5, far above eps 1e-8. So the scale changes nothing and the loss stays as the
  paper writes it (Σ_q c_q Σ_t).
- **The run:** `./training/h100.sh grpo RUN_NAME --cache-dir … --model-weight START [--gen-batch
  … --micro-batch …]` claims, locks, preflights and syncs exactly as `run` does
  ([sft-recovery](sft-recovery.md#launcher)). Its boot-hook line, and a restore, use `grpo` where
  an SFT run's use `run`: either subcommand on the other's run fails at argument parsing.
  - **Start weight:** `--model-weight` loads on every resume, before the pair replaces it, so it
    must outlive the run: keep it in `weights/`, not in an SFT run directory that gets cleared.
  - **Disk:** the preflight needs 75 GiB (56 with `--keep-checkpoints 1`), free plus what the run
    directory holds. An SFT run's pairs on the same disk (~81 GiB with fp32 moments, ~47 with 8-bit)
    go first, once they are on the Hub.
  - **Stopping:** `kill -TERM` the launcher's pid alone and wait for its final sync (~5 min): it
    uploads the newest pair, the best and the logs, which reach the Hub only then. A group kill
    (`kill -- -PGID`, `restart.sh`) kills `hf_sync` too. Nothing restarts a crashed run.
  - Relaunching a finished run loads the model and exits 0 (`train.py` exits before loading).
  - No commits or edits to tracked files in the run's worktree while it runs: a resume needs
    `run_id.json`'s commit at HEAD, and the preflight no tracked changes.

  `train_log.jsonl` holds, per update:
  - the loss, the ratio mean and its maximum deviation, and the clip fraction;
  - the gradient norm and the final BLEU;
  - the window's generation, reward-wait and old-logprob seconds.

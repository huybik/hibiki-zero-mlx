# GRPO

Read when touching `training/finetune/grpo.py`, `grpo_profile.py`, `rewards.py` or
`h100.sh grpo`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Checks](#checks) · [Running](#running)

The stage after SFT ([plan](../grpo_plan.md)): the paper's GRPO with BLEU process rewards, on
streams of the noisy `en_speech` cache. Checkpoints, resume and logs are the SFT run's
([sft-recovery](sft-recovery.md)), and evaluation is SFT validation's free-running pass
([sft-evaluation](sft-evaluation.md#checkpoint-selection)).

## Design

- **A window** is `--tau` (20) updates. Each window:
  1. a bf16 copy of the policy samples `--group` (4) rollouts of `--frames` (T, 1,200) frames for
     each of the window's `--streams` (32) × 20 streams;
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
  - Logprobs are taken at the sampling temperature 0.8; top-k is ignored.
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
- **Evaluation** runs at update 0 and every `--eval-every` (200) updates, as SFT validation's
  five sets. Promotion goes to the best FLEURS dev chrF among checkpoints that pass the gates and
  whose FLEURS median end offset is no later than update 0's.
  - That baseline is stored as `start_end_offset_s` in `run_config.json`, which `hf_sync` uploads
    whenever it changes. The logs reach the Hub only at the end, so a restore needs this file.
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

Not yet run on CUDA. Phase 1 and 2 profiles and the dry run go here.

## Running

- **Profile first, on the pod**, with `nvidia-smi` sampled beside it:
  `python -m training.finetune.grpo_profile --cache-dir … --model-weight START --out-dir …`.
  - It times generation at each `--gen-batch`, the rewards serially and in workers, and the
    old-logprob pass and the update at each `--micro-batch`.
  - It holds the fp32 policy and AdamW's states resident throughout.
  - It times each generation size on a second call, after compilation. Each batch's CUDA graph
    capture stays in the time, as in a run.
  - An out-of-memory size is recorded and skipped, and `profile.json` is rewritten after each
    stage.
- **Profiled on the box** (2026-11-06, 700 W H100 SXM, from `tranche_epoch1b`'s final model, the
  tranche caches, T 1,200, fp32 policy and AdamW states resident):

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
  that model was 26.2. The profile's GPU sat idle 39% of its 5.3 min, between its stages.
- **The run:** `./training/h100.sh grpo RUN_NAME --cache-dir … --model-weight START [--gen-batch
  … --micro-batch …]` claims, locks, preflights and syncs exactly as `run` does
  ([sft-recovery](sft-recovery.md#launcher)). Its boot-hook line, and a restore, use `grpo` where
  an SFT run's use `run`: either subcommand on the other's run fails at argument parsing.
  `train_log.jsonl` holds, per update:
  - the loss, the ratio mean and its maximum deviation, and the clip fraction;
  - the gradient norm and the final BLEU;
  - the window's generation, reward-wait and old-logprob seconds.

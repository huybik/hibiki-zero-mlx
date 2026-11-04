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
- **An update** takes its own 128 rollouts in `--micro-batch` forwards. Every rollout is used by
  exactly one update.
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
  - y_t joins the references of the rows whose Vietnamese has started by frame t, the row being
    spoken included. A row's reference is its cache text stream, decoded.
  - Reward frames fall every 31 frames (2.5 s) of Vietnamese speech, plus frame T, because the
    cache has no Vietnamese words.
  - Normalization uses the group's unbiased std plus 1e-4, so a frame where every rollout scores
    the same gets 0.
- **Evaluation** runs at update 0 and every `--eval-every` (200) updates, as SFT validation's
  five sets. Promotion goes to the best FLEURS dev chrF among checkpoints that pass the gates and
  whose FLEURS median end offset is no later than update 0's.
- **Saves** land on window boundaries (`--save-every` 20 updates). A resume starts its window
  again with the same seeds.

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
  - An out-of-memory batch is recorded and skipped.
- **The run:** `./training/h100.sh grpo RUN_NAME --cache-dir … --model-weight START [--gen-batch
  … --micro-batch …]` claims, locks, preflights and syncs exactly as `run` does
  ([sft-recovery](sft-recovery.md#launcher)). `train_log.jsonl` holds, per update:
  - the loss, the ratio mean and its maximum deviation, and the clip fraction;
  - the gradient norm and the final BLEU;
  - the window's generation, reward-wait and old-logprob seconds.

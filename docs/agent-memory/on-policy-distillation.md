# On-policy distillation

Read when touching `grpo.py`'s teacher term (`--teacher-weight`, `--reward-coef`, `--kl-coef`,
`distill_loss`, `update_loss`) or its `--advantage`, or when running, profiling or judging the
phone student's on-policy runs and the recipe experiments of 2026-11-21.
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Design](#design) · [Running](#running) ·
[Experiments](#experiments) · [Checks](#checks)

GRPO itself is in [grpo](grpo.md); the student's off-policy distillation in
[phone-student](phone-student.md#distillation); the plan in
[vi_en_parity_plan](../vi_en_parity_plan.md#5-the-phone-model), stage 5.5.

## Why

The students distil off-policy: each token's KL(teacher ‖ student) on the dataset's streams,
teacher-forced. Two of that recipe's known weaknesses show in their logs (user, 2026-11-21:
"update the current training code with new plan and experiments"):

- The student never sees its own mistakes: teacher-forced loss falls while free-running
  translation lags, as in the early SFT runs ([CONTEXT](../../CONTEXT.md#where-things-stand)).
- Forward KL covers the teacher's whole distribution, so a small student spreads its mass and
  sampling adds words: `student_dep10_distill`'s VI→EN length ratio rose 1.51 → 1.67, and the heads
  student at 96,000 writes 1.42–1.62 times the reference's length on the VI→EN sets against
  1.08–1.27 on EN→VI ([student-heads](student-heads.md#runs)), where its gap to the teacher is
  widest (FLEURS dev chrF 30.3 against 42.7).

On-policy distillation (Agarwal et al. 2024, arXiv 2306.13649; Thinking Machines, "On-Policy
Distillation", 2025) has the student generate and the teacher score the same rollouts, and
minimizes the reverse KL(student ‖ teacher) there: mode-seeking, and corrected where the student's
own rollouts go, timing included. 2026 work finds it teaches behaviour, not knowledge
(arXiv 2610.09639), so it follows the off-policy distillation rather than replacing it.

## Design

`grpo.py --teacher-weight` (and `--teacher-config`, by default the 3B's) loads a frozen bf16
teacher (`common.load_teacher`, the same as SFT's distillation) on the run's tokenizer.

```text
window: π_old samples rollouts (as GRPO) ──► rewards in the CPU workers (logged; the GRPO term's)
update micro-batch: teacher forward (no grad) ─┐
                    policy forward ────────────┴─► reward_coef · GRPO term + kl_coef · KL term
```

- **The term** (`distill_loss`): each kept token's reverse KL(policy ‖ teacher) of its full
  next-token distribution at temperature 1 (`common.token_kl` with its arguments swapped), where
  both forwards and `kept_tokens` hold it; weighted as the GRPO term is (`--text-weight` for
  text, 1 for each codebook), summed over tokens and averaged over the update's rollouts. The
  full distribution, not the sampled token's log ratio, so no variance from sampling. Rollouts are
  sampled at evaluation's temperature 0.8 and top-k 250, the deployed sampler.
- **One forward of the policy** serves both terms (`update_loss`). `--reward-coef 0` is
  distillation alone: no π_old pass and no ratio statistics; the rewards still run in the workers,
  for the log's `final_bleu`. Both terms on (the defaults, `--kl-coef 1`) keep a GRPO run near its
  teacher; a teacher needs a positive `--kl-coef`. With both, `--kl-coef` sets their balance, and
  centered advantages (in BLEU points) weigh the GRPO term differently from z-scored ones, so the
  coefficient does not carry over between the two.
- **The teacher must read the policy's codebooks**: it scores the rollout, which holds only the
  policy's. The 16-codebook heads student fits the 3B; the 8-codebook depformer student does not
  (the 3B reads 16 a stream).
- While a window generates, the teacher's parameters wait in pinned host memory beside the
  policy's and AdamW's (`HostOffload`): generation gets the GPU memory a GRPO run gives it. Each
  update micro-batch then runs the teacher's compiled training forward without a backward.
- Its delays may differ from the policy's (a heads student's codebook j at min(j, 8), the 3B's at
  2): a token is kept only where both forwards score it (`scored_delays`: each codebook's larger
  delay, for `kept_tokens` and the forward lengths alike).
- **The caveat**: unlike SFT's distillation, whose context is the dataset's, the teacher here reads
  the policy's samples, some of them sampled after the token it scores and conditioned on it. Its
  distribution can then lean towards the policy's own choice, so the KL reads low and partly
  rewards the policy's samples. With the heads student under the 3B (PR #149's reviews, on tiny
  random models):
  - Through the main transformer: the 3B reads codebooks 3–15 at delay 2, the student samples them
    at min(j, 8), so at text frame t the 3B reads frames t − min(j, 8) … t − 3 of them, sampled from
    step t on (codebook 8 at frame 17 moved its text logits at frame 20 by up to 0.26); at an audio
    token likewise (codebook 8 at frame f − 1, sampled at step f + 7, moved its codebook-2 logits at
    frame f by 0.08).
  - Through the 3B's depformer: it scores codebook 1 of frame f after the text and codebook 0 of
    frame f + 2, which the student samples a step after that codebook (0.16).
  How much the real teacher leans this way is unmeasured; `grpo_profile.py` bounds all but the
  depformer's part (`teacher_leak`, [Running](#running)).
- The train log adds `kl_text` and `kl_audio`, the mean unweighted KL a token; the ratio
  statistics appear only with the GRPO term.
- **`--advantage centered`** (Dr. GRPO, arXiv 2503.20783): each reward frame's advantage is
  r − mean over the group, in BLEU points, without the paper's division by the std
  (`rewards.normalize`); with the std a frame where the rollouts differ by one n-gram weighs as
  much as one where they differ widely. The loss already sums tokens without a per-rollout
  length normalization, Dr. GRPO's other change.

## Running

Profile first ([CONTEXT](../../CONTEXT.md#essential-invariants)): `grpo_profile.py` takes the same
`--teacher-weight`, `--reward-coef`, `--kl-coef` and `--advantage` and times the update with the
teacher's forward, the teacher parked while generating. With a teacher it also reports
`teacher_leak`. Each codebook the teacher delays less than the policy is shifted later in the
teacher's input by the difference (the heads student's codebook j by min(j, 8) − 2), so at every
token the teacher reads only frames the policy had sampled by then. The teacher's mean logprob of
the policy's kept tokens as sampled, minus shifted, is the caveat's bias plus what the misaligned
codebooks cost the teacher (taken as never negative): an upper bound, for text and each codebook
(`text_bound_nats`, `audio_bound_nats`), on the first 8 rollouts. Codebook 1's depformer part is
not in it. Against the start's `kl_text` and `kl_audio` (the micro-batch entries): well under
them, go ahead; near them, look closer before the run, where the shifted input is a leak-free
teacher at the misalignment's price. Sizes then go to `grpo_launch.sh`'s ARGS, after the 3B's:

```bash
.venv/bin/python -m training.finetune.grpo_profile --out-dir runs/opd_profile \
  --cache-dir <grpo_launch.sh's caches> --model-weight weights/<student start>.safetensors \
  --config-path weights/student_heads14_cb16/config.json \
  --teacher-weight weights/grpo_twoway_best1400.safetensors --reward-coef 0
training/tranche/box/grpo_launch.sh opd_heads weights/<student start>.safetensors \
  --config-path weights/student_heads14_cb16/config.json \
  --teacher-weight weights/grpo_twoway_best1400.safetensors --reward-coef 0 \
  --lr <probed> --gen-frames <profiled> --micro-batch <profiled>
```

The heads student keeps `--text-weight` 100 (16 codebooks: the 3B's 100:16). Evaluation and
promotion are GRPO's ([grpo](grpo.md#design)): the latency gate holds a checkpoint to its start's
end offset plus 0.2 s.

## Experiments

Planned 2026-11-21, each on the box after the run before it (one GPU job at a time; the heads
run `student_heads14_cb16` holds the GPU until it stops). Every validation is recorded in full
([sft-evaluation](sft-evaluation.md#recording-results)).

0. **Validation first** (user, 2026-11-21: "after the heads run and before grpo"): once the heads
   run stops, before any GRPO-family run, the new code runs end to end on the box on short runs:
   the profile with the teacher (memory, the update's speed, `teacher_leak`); a short
   `--reward-coef 0` run through an evaluation, a kill and a resume; the same with both terms and
   `--advantage centered`; plain GRPO's first updates against main's log; a `train.py --schedule
   constant` run and a `decay` branch from its pair; `average.py` on two of its checkpoints; a
   short `train.py --sampled-text 0.1` run through a validation, a kill and a resume
   ([exposure-bias](exposure-bias.md#checks)).
1. **On-policy distillation of the heads student**, distillation alone (`--reward-coef 0`), from
   the heads run's promoted checkpoint when it stops, teacher GRPO update 1,400. First the profile
   and its `teacher_leak` bound ([Design](#design)'s caveat), then a rate probe: 200 updates each at 1e-6 (the student GRPO's rate), 3e-6 and 1e-5 (the off-policy
   anneal's peak), judged by the KL's fall and the evaluation at 200. Then the run, stopped by
   the user's rule (two validations without a new best). It wins if, per GPU-hour, its selecting
   mean chrF beats the off-policy run's continuation, with the length ratio falling and no later
   end offset.
2. **The student's GRPO with the teacher's term** (`--kl-coef` > 0) against without, from the
   same start, both on the heads student (the depformer student cannot take the 3B as teacher).
   The motive is `grpo_student`, the depformer student, which collapsed between updates 2,200 and
   2,400 (length ratio 3.0, [student-grpo](student-grpo.md#evaluations)); the heads student's
   plain GRPO has not run yet, so whether it collapses too is part of the test. Judged by the best
   selecting mean and whether either degrades past 2,000.
3. **Centered advantages** against z-scored ones: two GRPO runs from one start and seed, 800
   updates each; the selecting mean and the length ratio decide.

Warmup-stable-decay and checkpoint averaging for SFT runs are in
[sft-training](sft-training.md#schedules-and-averaging); training against the phone's
palettization is stage 6 of the plan.

## Checks

- Tiny models on the CPU (2026-11-21; dim 64, a 16-codebook heads student under a depformer
  teacher with delays 0 and 2, eager with `lm_training.sdpa_kernel` replaced; scratch script):
  - `distill_loss` on a rollout the student sampled through `grpo.rollout` equals a hand-computed
    reverse KL on the positions both masks and `kept_tokens` hold (831.5887 against 831.5887),
    drops the tokens after a rollout's first EOS, and scores codebook j up to the later of the
    two models' delays; a teacher equal to the policy gives exactly 0; gradients are finite and
    reach only the policy.
  - `update_loss` with the GRPO term alone equals main's `grpo_loss` bit for bit; with both terms
    it is their weighted sum; `--reward-coef 0` needs no π_old.
  - The review (2026-11-21, its own tiny models): plain GRPO's loss, gradients and log lines are
    main's; no NaN reaches the loss or the gradient through either argument of `token_kl`, in eval
    mode and in the training forward run eagerly; `HostOffload` dedupes the teacher's storages
    and moshi's LM has no buffers to move.
  - `rewards.normalize` z-scored equals main's; centered is r − mean; an even frame gets 0.
- Not yet run on CUDA: the teacher's compiled forward in the update loop, its memory, and the
  offload with a teacher. The profile above is the first CUDA run.

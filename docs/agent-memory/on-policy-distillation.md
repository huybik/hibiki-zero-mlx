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
- While a window generates or an evaluation runs, the teacher's parameters wait in pinned host
  memory, beside the policy's and AdamW's while it generates and AdamW's while it evaluates
  (`HostOffload`): generation gets the GPU memory a GRPO run gives it. Each update micro-batch then
  runs the teacher's compiled training forward without a backward.
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
  --cache-dir <grpo_launch.sh's caches> --pair-cache-dir <its pair cache> --model-weight weights/<student start>.safetensors \
  --config-path weights/student_heads14_cb16/config.json \
  --teacher-weight weights/grpo_twoway_best1400.safetensors --reward-coef 0
training/tranche/box/grpo_launch.sh opd_heads weights/<student start>.safetensors \
  --config-path weights/student_heads14_cb16/config.json \
  --teacher-weight weights/grpo_twoway_best1400.safetensors --reward-coef 0 \
  --lr <probed> --gen-frames <profiled> --micro-batch <profiled>
```

The heads student keeps `--text-weight` 100 (16 codebooks: the 3B's 100:16). Evaluation and
promotion are GRPO's ([grpo](grpo.md#design)): the latency gate holds a checkpoint to a 3 s end
offset on each FLEURS dev set (until 2026-11-25, its start's plus 0.5 s).

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
   short `train.py --sampled-text 0.1` run through a validation, a kill and a resume. Done
   2026-11-21, all passed ([Checks](#checks), [exposure-bias](exposure-bias.md#checks)).
1. **The heads student's on-policy stage, aimed at the best phone model** (user, 2026-11-21: "i'm
   not asking for plain grpo, i'm asking for best outcome"): arms run only as probes, and only the
   winner runs long. Teacher GRPO update 1,400, one seed. The first probes (distillation alone and
   plain GRPO) start from the heads run's promoted checkpoint (176,000); both terms and the run
   from its final update (190,126, `weights/student_heads14_cb16_final.safetensors` on the box;
   user, 2026-11-22: "we can use checkpoint 190,126 as the mean increased": validation's seven-set
   mean 37.72 against 37.41, the test sets tie, 39.81 against 39.87):
   - **Probes**, 200 updates each, judged against their own update 0 by the evaluation at 200 (the
     selecting mean, the length ratio, the end offset) and the KL's fall. Distillation alone
     (`--reward-coef 0`) at 1e-6 (the student GRPO's rate), 3e-6 and 1e-5 (the off-policy anneal's
     peak). Then both terms (`--kl-coef` 1) at the best of those rates, z-scored and `--advantage
     centered`. Plain GRPO at 1e-6 is only the control that says whether the teacher's term helps;
     it gets no run of its own.
   - **The run**: the probes' winner, stopped by the user's rule (two validations without a new
     best). When both terms and distillation alone are close at 200, both terms run, since they
     also climb the reward evaluation measures. A collapse (the depformer student's plain GRPO
     reached length ratio 3.0 between updates 2,200 and 2,400,
     [student-grpo](student-grpo.md#evaluations)) or an end offset past the gate disqualifies a
     checkpoint.
   - **Then the phone's finish**: the run's best trained against the 6-bit palettization
     (`train.py --palettize-bits 6`, plan stage 6.4). Done 2026-11-25 from `both_heads_z`'s 1,200:
     its test chrF equals the start's and the start's snapped without training, within noise, and
     on the Neural Engine it agrees with bf16 on 0.9 points more tokens than the plain 6-bit export
     ([student-palettize](student-palettize.md#neural-engine-agreement)).
   - **Where it stands** (2026-11-25): the probes are done and `both_heads_z` (both terms,
     z-scored) won ([heads-on-policy-runs](heads-on-policy-runs.md#probe-results)); `both_heads_z2`, its
     recipe on the new data, went on to 1,200 with a 3 s latency limit (best 800, test mean of 11
     sets 41.18, below `both_heads_z`'s 200 at 41.56; its VI→EN drifts into Vietnamese;
     [results](heads-on-policy-runs.md#both_heads_z2)); `both_heads_z` continued to 1,800 (best
     1,200, test mean of 11 sets 42.53 against the start's 39.57;
     [continued](heads-on-policy-runs.md#both_heads_z-continued)), the best heads student so far.

Warmup-stable-decay and checkpoint averaging for SFT runs are in
[sft-training](sft-training.md#schedules-and-averaging); training against the phone's
palettization is stage 6 of the plan.

The probes' results and the runs, `both_heads_z` and `both_heads_z2`, are in
[heads-on-policy-runs](heads-on-policy-runs.md).

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
- **The box validation** (RTX PRO 6000, 2026-11-21, item 0; the heads student's best at 176,000
  under GRPO update 1,400; `/workspace/val_stack.sh`, logs in `/workspace/logs/val/`). All passed:
  - **Profile with the teacher** (`--reward-coef 0`): generating 512 rows of 1,200 frames 47.2 ms a
    step (36.0 GiB peak), 1,024 rows 91.1 ms (69.6 GiB), so a first window's update takes 10.7 s
    and 11.2 s; the update at micro-batch 2, 4 and 8 5.2, 5.0 and 5.8 s (2 peaked at 31.0 GiB; the
    others' peaks were inflated by the size before's graph until 2026-11-24).
    The sizes: `--gen-frames 614400 --micro-batch 4`, ~16 s an update with generation. Plain
    GRPO's profile picks the same (~14 s an update); its first update's ratio deviation is 0.
  - **`teacher_leak`**: text 0.006 nats against `kl_text` 0.08, codebooks 0–2 at most 0.11; 3–15
    1.1–2.2 against `kl_audio` 1.25–1.36. There the shift (up to 6 frames) misaligns each codebook
    with the lower ones of its own frame, which costs the teacher more than any leak could, so
    the bound says nothing for them. Text, which chrF and the gate score, is clean.
  - **Plain GRPO, main against this branch** (6 updates, `--tau 2`, the same start and seed): the
    same evaluation at 0, update 1's loss −10114.0 in both, grad norms 155,786 and 155,773; the
    rollouts part from update 3 (the backward is not bit-exact); FLEURS dev 37.69 / 36.03 and
    37.72 / 35.98 after 6.
  - **Distillation alone** (1e-6, 40 updates, `--tau 10`): SIGKILLed right after the update-20
    pair, mid-evaluation; the restart resumed at 20, evaluated 20 again and ran to 40, each update
    logged once. `kl_audio` 2.10 → ~1.4 and `kl_text` 0.098 → ~0.08. FLEURS dev 38.12 / 36.10 →
    39.31 / 36.50 (20) → 39.74 / 36.56 (40), VI→EN length ratio 1.31 → 1.23, but EN→VI's end
    offset 1.4 → 1.9–2.0 s and its rows ending with EOS 193 → 88 of 392: the latency gate held both
    checkpoints.
  - **Both terms, `--advantage centered`** (20 updates, killed after the update-10 pair): resumed
    and promoted at 10 and 20, FLEURS dev 38.62 / 36.39 at 20 with the start's end offsets. The
    GRPO term's gradient (norm ~7e5) dwarfs distillation's (~3e4): `kl_audio` stayed 1.7–2.7.
  - A student pair is 15 GB (model 5.1, AdamW's 10.2): short runs on this 300 GB disk keep one.

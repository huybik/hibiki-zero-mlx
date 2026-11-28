# Exposure bias and the 3B's recipe

Read when choosing between training a model from scratch and fine-tuning the 3B, when
considering scheduled sampling or another fix for exposure bias in SFT, or when running or judging
the 3B's recipe experiments of 2026-11-21 (scheduled sampling, GRPO anchored to its start).
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Scratch or fine-tune](#scratch-or-fine-tune) ·
[Scheduled sampling](#scheduled-sampling) · [Anchored GRPO](#anchored-grpo) ·
[Experiments](#experiments) ·
the runs and their full tables: [exposure-bias-runs](exposure-bias-runs.md)

The student's own recipe experiments (on-policy distillation, centered advantages) are in
[on-policy-distillation](on-policy-distillation.md); the plan in
[vi_en_parity_plan](../vi_en_parity_plan.md), stages 3 and 4.

## Why

SFT is teacher-forced: at every frame the model reads the dataset's text and codebooks and
predicts the next. Generation reads its own samples instead, at temperature 0.8 and top-k 250,
so in training it never sees its own mistakes or timing (user, 2026-11-21: "should we change
teacher enforce"). The gap shows in every run here:

- Teacher-forced loss falls while free-running translation stays poor: the PhoMT run
  ([CONTEXT](../../CONTEXT.md#where-things-stand)), run 1's validation loss 2.73 → 2.67 with
  content accuracy flat at 0.63 and FLEURS dev chrF 24.9 at its end
  ([plan](../vi_en_parity_plan.md#order)).
- On-policy training moves what teacher forcing doesn't: after two epochs of two-way SFT (287,558
  updates, 2026-11-11 to 13), GRPO's 1,400 updates (~8 h) took the 11 test sets' mean chrF
  41.12 → 45.48 and FLEURS's end offset 2.7 → 0.9 s VI→EN ([grpo](grpo.md#running)).

## Scratch or fine-tune

Decided 2026-11-21 (user: "if we train vi - en and en-vi from scratch or finetune from the 3b,
how the plan recipe should change"): **fine-tune**, as the plan does
([plan](../vi_en_parity_plan.md#decisions-2026-11-01-later-dates-marked)).

- A random start must learn speech and Mimi's codes before translating. The family's speech
  comes from pretraining on audio far past our ~8,500 h a direction (Moshi, its base,
  pretrained on ~7M hours), and the text side from a pretrained text LM. Our evidence: a student
  cut from the 3B's own layers fell ~14 chrF short before distillation
  ([phone-student](phone-student.md)); a random start keeps not even those layers. The same
  holds for a phone-sized model: cut it from the 3B and distil.
- The only "from scratch" worth costing is a restart from upstream's 3B (through the tokenizer
  swap's `init_weights`) in one two-way SFT stage on all the data, in place of the chain (run 1 →
  run 2 → the tranche → round 4 → two-way → GRPO). Not now: it costs at least the two-way run's
  ~2 days, and each new stage so far cost something first (the learning-rate test's and round
  4's dips, two-way VI→EN 41.3 → 39.6). It pays when the data changes enough to retrain on: the
  tranche's scale-up toward ~10k h of real Vietnamese ([plan](../vi_en_parity_plan.md#1-data),
  1.3). Then it runs warmup-stable-decay, its decay branches choosing how much data
  ([sft-training](sft-training.md#schedules-and-averaging)), and GRPO after it.

What changes in the recipe either way:

```text
SFT, teacher-forced       (scheduled sampling of the words tied it: not adopted)
cosine per stage     ──►  warmup-stable-decay, averaging
GRPO, rewards only   ──►  GRPO + KL to its start         (anchors what no reward reads)
```

Teacher forcing stays the bulk objective: it is the only one that covers ~17k h, every frame of a
batch at once (14–15k frames/s on the H100), where generation is one frame at a time (GRPO's
~6–8 rollouts/s). The on-policy part sits on top of it.

## Scheduled sampling

Parallel scheduled sampling of the words (Duckworth et al. 2019, arXiv 1906.04331; for
transformers, Mihaylova and Martins 2019) **tied plain SFT at rate 0.1 and was removed**
([Experiments](#experiments) item 1); its code is `train.py --sampled-text` (`sample_text`) and
`lm_training.forward_history` at commit 364091a. Its design, for anyone who retries it:

```text
pass 1, no grad:  main transformer(codes) ─► sample every scored text token (0.8, top-k 250)
mix:              history = codes; each word piece ← the sampled word piece, with probability P
                  (PADs, the EOS and a word sampled as PAD stay the dataset's)
pass 2, grad:     main transformer reads history (earlier steps) ┐
                  depformer reads codes (this step's own tokens) ┴► loss against codes
```

- **Words, not timing.** The targets stay the dataset's, and 76–79% of text frames are PAD, where
  a sampled token that differs is mostly a word a frame early or late: one sampled early would
  teach the model to write it again a frame later, one sampled late to drop it. So only a word
  piece sampled as another word piece is replaced. Timing, the EOS and length are GRPO's.
- **The text only.** Codebooks sampled each on its own in one parallel pass would pair a sampled
  codebook with the dataset's next, a frame no audio has (Mimi's residual chain); sampling the
  audio means 16 depformer passes in a row a step, as `LMGen` draws them. GRPO already trains on
  the model's own speech.
- **What stays inconsistent**: the step that first reads a sampled word also reads the dataset's
  audio of that frame, so the model can recover the dataset's word from the dataset's speech, a
  cue generation never gives.
- **Cost** on the RTX PRO 6000 (the 3B, 4,608-frame batches): 0.49 s a step against 0.43, 75.3 GB
  against 74.8 at the peak; not with a teacher, whose targets are its distributions on the
  dataset's stream (a student learns on-policy in `grpo.py`).

## Anchored GRPO

`grpo.py --teacher-weight <its start> --kl-coef β` adds each kept token's reverse
KL(policy ‖ start) to the 3B's GRPO: the KL penalty to a reference model of RLHF and of GRPO as
published (DeepSeekMath used β = 0.04), built for the student's on-policy distillation
([on-policy-distillation](on-policy-distillation.md#design)), here with the 3B as its own teacher.

- Same model, same delays and codebooks: none of that note's leak (`grpo_profile.py` reports no
  `teacher_leak`). The term starts near 0, not at it: the policy's load copies the text PAD's
  input row over EOS's ([sft-training](sft-training.md#trainer), `model.load_policy`) and the
  teacher keeps the file's, so the few kept tokens that read an EOS (codebooks 1–15 of the frame
  before it) differ (KL 0.01–0.06 on a tiny model), and fp32 masters differ from a bf16 copy a
  little everywhere. Read the profile's starting `kl_text` and `kl_audio` as that floor.
- The motive: the shipped run's selecting mean stopped rising after update 1,400 while its EOS
  share fell 0.83 → 0.68, a drift no reward checks (no reward pays for an EOS, the audio is
  scored only through its text) ([grpo](grpo.md#running)). The anchor holds the policy near the
  SFT model wherever the rewards say nothing, and may let a run improve longer.
- Memory: the start's bf16 copy (5.8 GiB) waits in pinned host memory while a window generates and
  joins the update's micro-batches on the GPU; the shipped run's update peaked at 68 GiB at
  micro-batch 2 on an 80 GB H100. Profile with it first (`grpo_profile.py` takes the same flags).
- Not for the heads student (2026-11-23, user: "should we add that the both_heads_z?"): its
  teacher term is this anchor, toward a better model than its own start (selecting ~40.8 against
  37.33), and without it plain GRPO wrecked the student in 200 updates
  ([heads-on-policy-runs](heads-on-policy-runs.md#probe-results)). Its EOS share falls with
  distillation alone as much (FLEURS-en dev 191 → 89 of 392), so a second KL would not hold it,
  and the runtimes never stop at an EOS. If its end offset drifts past the gate, raise
  `--kl-coef` instead.

## Experiments

Planned 2026-11-21, after the student's experiments, each on the box after the run before it.
From 2026-11-23 experiments 1 and 2 open the teacher's next round, before its full SFT
([teacher-next](teacher-next.md#order)). Every validation and test is recorded in full
([sft-evaluation](sft-evaluation.md#recording-results)).

1. **Scheduled sampling in SFT.** Two continuations of the two-way SFT's final model on the next
   round's caches ([teacher-next](teacher-next.md#data); a new `--seed`, the same for both),
   10,000 updates each, 5% warmup to 1e-6 (the learning-rate test's
   rate that stayed within noise, [plan](../vi_en_parity_plan.md#order) item 4) and cosine to 0:
   rate 0 against 0.1. Judged at the end on FLEURS dev both ways and `en-vi-val`
   (chrF, length ratio, end offset, rows ended). GRPO fixes the same gap, so a win counts only if
   it survives GRPO: 400 updates from each arm's final model at the shipped run's settings,
   compared on validation's selecting mean. Validation selects on the text and reports only the
   voice scores ([voice-scores](voice-scores.md)), and the audio is trained
   after the mixed text too, so before adopting the winner its generated speech is compared with the control's
   (`eval.py` without `--text-only` writes it): an ASR transcript of each arm's English on FLEURS
   dev scored against the reference, as stage 7.1 measured Mimi, and a listen; that script is
   written with the experiment. If 0.1 wins all three, try 0.25. The control arm is also the
   round's first look at its data: after its SFT EN→VI should rise and VI→EN hold, and its GRPO
   at 400 compares with the shipped run's (selecting 38.89), read as each run's change from its
   own update 0, since `grpo_launch.sh`'s caches gained the noisy EN→VI rows on 2026-11-23.
   10,000 updates read ~670 h, so it catches data that hurts rather than proving that a full
   pass helps; if it hurts, the full SFT doesn't run.
   **Run 2026-11-23** (GRPO cut to 200 updates an arm, the rental ending;
   [tables](exposure-bias-runs.md#scheduled-sampling)): **0.1 is a tie, so it isn't adopted** (the
   option is removed) and its speech check and 0.25 don't run (0.1 − control, selecting: −0.23 [−1.06, +0.68] after the SFT,
   +0.59 [−0.20, +1.42] after GRPO). The data gate **fails on VI→EN FLEURS dev** (−3.85 [−5.54, −2.30]
   against `twoway_final` at seed 44, −3.66 at 61; EN→VI isn't clearly up), and GRPO doesn't recover
   it. The loss is on quiet Vietnamese clips, where the new model depends on the input's level and
   `twoway_final` doesn't ([the quiet clips](exposure-bias-runs.md#the-quiet-clips); one model, cause
   not established): the user decides whether the full SFT runs.
2. **Anchored GRPO.** From the shipped run's start, the two-way SFT's final model, on its streams
   and seed (its `run_config.json`'s caches, given after `grpo_launch.sh`'s, which changed on
   2026-11-23): first a probe of 200 updates each at β 0.02 and 0.1, judged by the KL's size
   against the GRPO term and the evaluation at 200 against the shipped run's (selecting 38.00);
   then the run, against the shipped run's evaluations ([grpo](grpo.md#running): selecting mean
   40.76 at 1,400). It wins with a higher best selecting mean, or the same with a higher EOS
   share and no later end offset; the test sets decide between the two runs' best checkpoints.
   In the teacher's next round the probe alone runs first and decides whether that round's GRPO
   is anchored and at which β; that GRPO is then the run, whose comparison with the shipped run
   also carries the new SFT.
   **Probed 2026-11-23** ([tables](exposure-bias-runs.md#anchored-grpo)): at 200, β 0.02 reads 39.12
   selecting (+1.12 [+0.44, +1.88] against the shipped run's 38.00) and 0.1 reads 38.31 (+0.31 [−0.46,
   +1.13]); the mean of the eight sets +0.62 and +0.63 for both. 0.02 is ahead on selecting (by 0.81
   [0.12, 1.60], about the noise between two runs of one model) and even on the mean of eight, so
   they aren't separated; if a round anchors, 0.02. Not proved: the KL per
   token stays ≤ 0.005 and its share of the update is unmeasured, the EOS share still fell (82.6% →
   79.0% at 0.02), and no plain GRPO from the same start on today's code was run to isolate the anchor.
3. **Centered advantages on the 3B**: dropped (2026-11-23), and their option removed. They were
   to run only if they won on the student, and z-scored won there
   ([heads-on-policy-runs](heads-on-policy-runs.md#probe-results)).

The data stays the larger lever: the tranche's first 1,760 h moved every real-speech set by
9–10 chrF ([plan](../vi_en_parity_plan.md#order) item 6). These experiments tune the recipe that
the next data will be trained with.

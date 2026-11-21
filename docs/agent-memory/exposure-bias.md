# Exposure bias and the 3B's recipe

Read when touching `train.py --sampled-inputs` (`sample_inputs`), when choosing between training
a model from scratch and fine-tuning the 3B, or when running or judging the 3B's recipe
experiments of 2026-11-21 (scheduled sampling, GRPO anchored to its start).
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Scratch or fine-tune](#scratch-or-fine-tune) ·
[Scheduled sampling](#scheduled-sampling) · [Anchored GRPO](#anchored-grpo) ·
[Experiments](#experiments) · [Checks](#checks)

The student's own recipe experiments (on-policy distillation, centered advantages) are in
[on-policy-distillation](on-policy-distillation.md); the plan in
[vi_en_parity_plan](../vi_en_parity_plan.md), stages 3 and 4.

## Why

SFT is teacher-forced: at every frame the model reads the dataset's text and codebooks and
predicts the next. Generation reads its own samples instead, at temperature 0.8 and top-k 250,
so in training it never sees its own mistakes or timing (user, 2026-11-21: "should we change
teacher enforce"). The gap shows in every run here:

- Teacher-forced loss falls while free-running chrF stalls: the PhoMT run
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
  cut from the 3B's own layers fell ~15 chrF short before distillation
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
SFT, teacher-forced  ──►  SFT + scheduled sampling      (cheap: one more forward a step)
cosine per stage     ──►  warmup-stable-decay, averaging
GRPO, rewards only   ──►  GRPO + KL to its start         (anchors what no reward reads)
```

Teacher forcing stays the bulk objective: it is the only one that covers ~17k h, every frame of a
batch at once (14–15k frames/s on the H100), where generation is one frame at a time (GRPO's
~6–8 rollouts/s). The on-policy part sits on top of it.

## Scheduled sampling

`train.py --sampled-inputs P` (parallel scheduled sampling, Duckworth et al. 2019,
arXiv 1906.04331; for transformers, Mihaylova and Martins 2019):

```text
pass 1, no grad:  model(codes) ─► sample every scored token (0.8, top-k 250)
mix:              history = codes, each of the model's own tokens ← its sample with probability P
pass 2, grad:     main transformer reads history (earlier steps) ┐
                  depformer reads codes (this step's own tokens) ┴► loss against codes
```

- **Which tokens**: the model's own rows (text, then its target codebooks), where pass 1 scores
  them (its masks: not -1, and not a delayed row's last frames, which no step reads). The source
  rows stay the dataset's. A replaced token becomes only an input, never its own target.
- **Only the history mixes** (`lm_training.forward_history`): the main transformer reads the
  mixed codes, the steps before each one, where exposure bias compounds; the depformer (a heads
  student's text input) reads each step's own tokens from the dataset. Within a step the codebooks
  form Mimi's residual chain: after a sampled codebook k − 1, the dataset's codebook k encodes
  another residual than the sample leaves, a conditional no audio has (PR #151's review). Across
  steps a sampled frame followed by the dataset's next one is scheduled sampling's intended
  recovery.
- **Sampler**: validation's (`common.TRAINER_GENERATION`), the runtimes' and GRPO's: temperature
  0.8 and top-k 250 for text and audio.
- **One rate** for text and audio. The audio errors are the sampler's own noise at every frame;
  the text errors are content and timing (a word early or late, another word). A replaced word
  keeps the dataset's next word as target, so the model learns to come back to the reference
  after its own deviation: on a paraphrase that teaches an abrupt join, so the rate starts low.
- **Logs**: the train log's losses read mixed inputs and run above a plain run's; compare runs by
  validation, which is unchanged (reference inputs, then free-running). `inputs_changed` is the
  share of the model's own scored input tokens that differ from the dataset's.
- **Cost**: one forward without gradients a step, about a third of the step (unprofiled). Its
  logits are sampled in float32: ~3 GB of logits and probabilities at 4,608 frames, text's 48,000
  entries most of it, freed before pass 2.
  Profile both arms before the runs ([CONTEXT](../../CONTEXT.md#essential-invariants)).
- **Not with a teacher** (refused): distillation's targets are the teacher's distributions on the
  dataset's stream; a student learns from its own samples on-policy in `grpo.py`
  ([on-policy-distillation](on-policy-distillation.md)). With `--palettize-bits` pass 1 runs on
  the palettized weights, as pass 2 does.

## Anchored GRPO

`grpo.py --teacher-weight <its start> --kl-coef β` adds each kept token's reverse
KL(policy ‖ start) to the 3B's GRPO: the KL penalty to a reference model of RLHF and of GRPO as
published (DeepSeekMath used β = 0.04), built for the student's on-policy distillation
([on-policy-distillation](on-policy-distillation.md#design)), here with the 3B as its own teacher.

- Same model, same delays and codebooks: none of that note's leak (`grpo_profile.py` reports no
  `teacher_leak`). The term starts near 0, not at it: the policy's load copies the text PAD's
  input row over EOS's ([sft-training](sft-training.md#trainer), `common.load_policy`) and the
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

## Experiments

Planned 2026-11-21, after the student's experiments, each on the box after the run before it.
Every validation and test is recorded in full ([sft-evaluation](sft-evaluation.md#recording-results)).

1. **Scheduled sampling in SFT.** Two continuations of the two-way SFT's final model on its
   caches (a new `--seed`, the same for both), 10,000 updates each, 5% warmup to 1e-6 (the learning-rate test's
   rate that stayed within noise, [plan](../vi_en_parity_plan.md#order) item 4) and cosine to 0:
   `--sampled-inputs 0` against `0.1`. Judged at the end on FLEURS dev both ways and `en-vi-val`
   (chrF, length ratio, end offset, rows ended). GRPO fixes the same gap, so a win counts only if
   it survives GRPO: 400 updates from each arm's final model at the shipped run's settings,
   compared on validation's selecting mean. Validation scores text only, and the mix reaches the
   audio too, so before adopting the winner its generated speech is compared with the control's
   (`eval.py` without `--text-only` writes it): an ASR transcript of each arm's English on FLEURS
   dev scored against the reference, as stage 7.1 measured Mimi, and a listen; that script is
   written with the experiment. If 0.1 wins all three, try 0.25.
2. **Anchored GRPO.** From the shipped run's start, the two-way SFT's final model, on its streams
   and seed: first a probe of 200 updates each at β 0.02 and 0.1, judged by the KL's size against
   the GRPO term and the evaluation at 200; then the run, against the shipped run's evaluations
   ([grpo](grpo.md#running): selecting mean 40.76 at 1,400). It wins with a higher best selecting
   mean, or the same with a higher EOS share and no later end offset; the test sets decide
   between the two runs' best checkpoints.
3. **Centered advantages on the 3B**, only if they win on the student
   ([on-policy-distillation](on-policy-distillation.md#experiments), experiment 3).

The data stays the larger lever: the tranche's first 1,760 h moved every real-speech set by
9–10 chrF ([plan](../vi_en_parity_plan.md#order) item 6). These experiments tune the recipe that
the next data will be trained with.

## Checks

- Tiny models on the CPU (2026-11-21; dim 64, a 3B-shaped model with delays 0, 0 and 2, and a
  16-codebook heads student, the training forward eager; scratch script):
  - `sample_inputs` at rate 0 returns the codes unchanged; at 0.5 about half the scored tokens of
    the model's own rows differ (0.485), never a source row, a -1 or an unscored position; its
    count equals the tokens that differ; near 1, 0.965 of them; a text head that favours one token
    by far samples only that token; a heads student runs through it.
  - `compute_batch_losses(..., history=mixed)` equals the CE of `forward_history(mixed, codes)`'s
    logits against the dataset's tokens, not against the samples; `history=codes` equals no
    history; gradients are finite.
  - `forward_history(codes, codes)` equals moshi's forward, on the 3B shape and the heads student.
    A token changed in the history alone (codebook 3 of frame 10) moves no prediction of its own
    step (codebook 4 of frame 10, which moves when the depformer reads it too) and nothing
    before it, and moves the next step's; a changed text token leaves its frame's codebook 0.
  - The review (PR #151, its own scripts): perturbing each own input in turn, on the 3B shape and
    a heads student, moves only predictions generation samples later; pass 1 on palettized
    weights restores the full-precision ones with finite gradients everywhere.
  - `validate_args` refuses a rate outside [0, 1) and a rate with `--teacher-weight`.
- Not yet run on CUDA: the compiled training forward without gradients (one more compile per
  shape) and `forward_history`'s, the step's time and memory. The validation before the student's GRPO
  ([on-policy-distillation](on-policy-distillation.md#experiments), item 0) adds a short
  `--sampled-inputs 0.1` run through a validation, a kill and a resume.

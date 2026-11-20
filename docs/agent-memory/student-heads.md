# Heads student

Read when touching `training/student/heads.py` or `restart.sh student-heads`, or training, scoring or
deciding on the phone student with parallel codebook heads. Back to [CONTEXT.md](../../CONTEXT.md) ·
[Design](#design) · [Codebooks](#codebooks) · [Speed](#speed) · [Runs](#runs). The cut, training and
distillation it shares with the depformer student are in [phone-student](phone-student.md).

## Design

`--heads` cuts the delay-pattern student (stage 5.4; the plan's is ~14 layers: `--num-layers 14`):
the kept main layers and `--codebooks`, no depformer, and `training/student/heads.py`'s
`ParallelHeads` (one linear head a codebook, initialized afresh with `--seed`) in the model's
`depformer` slot. The heads read the main transformer's normed output plus the model's own text
embedding of the step's text token (teacher-forced in training, the just-sampled token in
generation), as the depformer's first step reads that token: speech then follows the inner
monologue. Text is sampled before audio at each step, so this adds no latency and no parameters.

- Codebook j is delayed min(j, 8) frames (`heads.delays`, `DELAY_CAP`, written into the config's
  target delays; the source's keep the parent's): a head predicting codebook j ≤ 8 at a step has
  read codebook j-1 of the same frame in the step's input, as in MusicGen; codebooks 8–15 share
  delay 8 and each has read 0–7 of its frame, not each other ([Codebooks](#codebooks)). moshi's own
  delay handling then carries it through training (logits undelayed, masks) and generation
  (`LMGen` withholds the first max-delay steps: 7 at 8 codebooks, 8 at 16). The trunk also
  relearns its input timing: the parent delays target codebooks 1–15 by 2 frames, the heads
  student codebook j by min(j, 8). Its larger `max_delay` delays its speech, not its text, so
  evaluation's `end_offset_s` takes the extra frames off and the two students' text latency
  compares directly.
- In the `depformer` slot because generation samples audio only when a depformer exists:
  `heads.py` replaces `LMGen.depformer_step` (all codebooks sampled at once from the heads; any
  other model runs moshi's steps) and, per model, `forward_depformer_training`. The depformer's
  tensors are dropped; the config's `depformer_num_layers` is 1, the throwaway moshi builds first.
- The config marks it with `"audio_heads": "parallel"`; `common.load_checkpoint_info` takes the
  key out before moshi reads the config, and every model is built through `common.build_lm`
  (a heads model: built without weights, heads installed, then loaded, the EOS row copied from
  PAD's as moshi's own load does).

## Codebooks

16 a stream, codebooks 8–15 at one delay (user, 2026-11-15: "should we keep 16 codebook, since it's
parallel?"; 2026-11-20: "i remember we planned 16 codebook, with the later 8 sent as once or so to
not increase latency so much"). `restart.sh student-heads` cuts `student_heads14_cb16` with
`--codebooks 16`.

- A frame's speech completes 8 frames (640 ms) after its first codebook: 80 ms over the 8-codebook
  heads, 6 frames over the depformer student's 2. Fully delayed, 16 codebooks take 15 frames (1.2 s;
  rejected).
- For: 16-codebook speech (Whisper's Vietnamese WER through Mimi 27.8% against 42.0% at 8, English's
  6.3% against 8.9%) and 16 source codebooks to hear (gain unmeasured).
- Cost on the phone: 8 more heads (~34M parameters, under one main layer's ~63M) and 16 more
  embedding tables (~67M, one row read from each a frame): 1.279B parameters against 1.178B. The
  heads still sample in one pass.
- Untested: codebooks 9–15 are sampled without seeing 8..j-1 of their frame (MusicGen found fully
  parallel sampling worse). The teacher's depformer has read them, so on those codebooks the KL has
  a floor the student can't reach: it learns the teacher's distribution averaged over them.
- The change's review (2026-11-20, tiny CPU models with these delays): no logit moves unless its
  token is sampled after the changed one; greedy `LMGen` equals the teacher-forced argmax on text and
  all 16 codebooks; distillation scores where both masks hold (codebooks 8–15 lose their last 8
  frames); the end offset takes off `max_delay` − 2 = 6 frames.
- Losses don't compare with the 8-codebook student's: `audio` averages 15 acoustic codebooks, not 7,
  and carries that floor. Compare students by chrF, end offset and, once it exists, speech
  evaluation. Against the depformer student (8 codebooks) it changes both the heads and the codebook
  count.

## Speed

The RTX PRO 6000 (`bench.sh` on the box, the run's arguments, steps 151–300;
[sft-speed](sft-speed.md#by-gpu)):

| Codebooks | s an update | Frames/s | Peak memory |
|---|---|---|---|
| 8 | 0.564 | 20,700 | 44 GiB |
| 16 | 0.694 (+23%) | 16,900 | 50.9 GiB |

Profiled over 10 updates, 705 ms against 570: matmuls +71 ms, mostly the teacher's depformer, which
now runs all 16 steps (cut to the student's 8 before, [distillation](phone-student.md#distillation));
elementwise and copies +33 ms spread over many kernels (bf16 adds +17). The card stays 99% busy at
its 600 W cap; the one lever left is the FP8 teacher, declined ([sft-speed](sft-speed.md#by-gpu)).

## Runs

Both on the RTX PRO 6000 box ([tranche-box](tranche-box.md#state)) with `restart.sh student-heads`'s
arguments: the two-way run's rows both ways, 190,126 updates (95,063 batches an epoch), seed 47,
12,288-frame batches, peak 5e-5, distilled from GRPO update 1,400; validation every 16,000 updates,
free-running at seed 47, batches of 64.

- `student_heads14` (8 codebooks, `d37109e`, from 2026-11-20 01:22 UTC): stopped at ~22,600 for 16
  codebooks. Its pair and best at 16,000 are on the Hub (`student_heads14/`); its log stayed on the
  box (`/workspace/logs/student-heads-cb8.log`). Validation at 16,000 (2026-11-20; teacher-forced
  loss 3.505 / 3.310 on the VI→EN / EN→VI caches; selecting chrF 24.97):

  | Set | chrF | BLEU | Length ratio | End offset (s) | Health |
  |---|---|---|---|---|---|
  | fleurs-vi/dev | 26.75 | 2.15 | 1.55 | 2.4 | pass |
  | fleurs-en/dev | 23.18 | 5.78 | 1.10 | 2.4 | pass |
  | multimed-vi/select | 27.27 | 3.47 | 1.58 | 3.7 | pass |
  | en-vi-val/cv27 | 26.08 | 4.61 | 1.51 | 3.0 | pass |
  | en-vi-val/libriheavy | 31.33 | 6.11 | 1.18 | 3.0 | pass |
  | en-vi-val-en/cv27 | 22.00 | 6.01 | 1.20 | 0.7 | pass |
  | en-vi-val-en/libriheavy | 30.98 | 9.39 | 1.10 | 2.5 | pass |

  FLEURS dev chrF 26.75 / 23.18 against the plain depformer student's 20.7 / 15.0 at 16,000.
- `student_heads14_cb16` (16 codebooks, `ce80c0b`, from 2026-11-20 05:19 UTC, Hub
  `student_heads14_cb16/`): running, ~37 h plus 12 validations. Validations (teacher-forced loss on
  the VI→EN / EN→VI caches, not comparable with the 8-codebook run's; chrF / BLEU / length ratio /
  end offset in s, every set's health pass unless marked):

  | Update (date) | TF loss | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|---|
  | 16,000 (11-20) | 3.701 / 3.494 | 25.28 | 28.40 / 2.77 / 1.59 / 3.0 | 22.15 / 5.08 / 1.05 / 3.1 | 28.49 / 4.31 / 1.50 / 3.8 | 28.49 / 5.64 / 1.43 / 3.1 | 33.17 / 6.96 / 1.19 / 3.5 | 23.29 / 6.91 / 1.20 / 1.4 | 31.29 / 10.20 / 1.06 / 2.4 |

  At 16,000 against 8 codebooks: VI→EN +1.2 to +2.4 chrF on every set, EN→VI −1.0 (FLEURS) to +1.3,
  end offsets 0.1–0.7 s later; generation and scoring at 16 codebooks work on the GPU.

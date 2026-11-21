# Heads student

Read when touching `training/student/heads.py`, `training/student/palettize.py` or `restart.sh student-heads`, or training, scoring or
deciding on the phone student with parallel codebook heads. Back to [CONTEXT.md](../../CONTEXT.md) ·
[Design](#design) · [Codebooks](#codebooks) · [Speed](#speed) ([inference](#inference),
[training](#training)) · [Palettization](#palettization) · [Runs](#runs). The cut, training and
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

### Inference

The MLX runtime runs the heads ([mlx-inference](mlx-inference.md#vendored-moshi-mlx)); the Swift
runtime runs them as Core ML on the Neural Engine: on the A15 the LM takes 45–48 ms a frame, and
from cool the phone holds real time for 10 minutes; Mimi's CPU encode, not the LM, sets the margin
([lm-npu](lm-npu.md#speed)).
On the M4 Pro (2026-11-20; update 16,000 of
`student_heads14_cb16` against the shipped depformer student, GRPO update 2,000, both staged by
`convert_mlx.py`), the LM step alone (text and audio sampled, medians of 300 steps; scratch script):

| Model | Size at 6 bits | 6 bits, back to back | 6 bits, every 80 ms | bf16, back to back |
|---|---|---|---|---|
| Depformer student | 1.24 GB | 6.54 ms (main 3.56 + depformer 3.17) | 15.83 ms | 12.12 ms |
| Heads student, 16 codebooks | 1.16 GB | 5.20 ms (main 4.89 + heads 0.56) | 12.88 ms | 10.30 ms |

- The 16 heads in one pass cost a sixth of the depformer's 8 sequential steps, which pays for 4
  more main layers and 16 more codebooks. The two-way 3B at 6 bits: 19.56 ms back to back.
- Spaced as live frames arrive, the GPU clocks down between steps and each takes ~2x as long; the
  order holds. `bench.py` (Mimi on the CPU between steps) gives LM totals of 9.2–9.8 ms (heads) and
  10.1–11.5 (depformer) over runs; Mimi's encode (~18.5 ms) and decode (~17 ms) on the CPU are now
  the larger costs.
- The same update on the gate (6 bits, the first 20 FLEURS dev rows each way, `--seed 42`;
  `/Volumes/data/hibiki-zero-evals/mac/gate/heads_cb16_s16k_vs_dep10_g2000_q6_20rows/`): chrF / BLEU /
  length ratio / end offset / EOS, health pass on both:

  | Set | Heads, update 16,000 | Depformer student, GRPO 2,000 |
  |---|---|---|
  | fleurs-vi/dev (20) | 30.40 / 2.04 / 1.72 / 3.6 s / 15 of 20 | 43.67 / 7.95 / 1.14 / 1.2 s / 11 of 20 |
  | fleurs-en/dev (20) | 25.79 / 6.95 / 1.02 / 2.3 s / 17 of 20 | 35.09 / 15.04 / 1.02 / 1.6 s / 8 of 20 |

  Near the box's validation at that update on the whole dev sets (28.40 / 22.15), so MLX reads the
  heads student as training does; the gap is training (16,000 of 190,126 updates, no GRPO).

### Training

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

## Palettization

Training against the Core ML export's palettization (plan stage 6.4, 2026-11-21;
`training/student/palettize.py`, `train.py --palettize-bits`): on the Neural Engine the 6-bit
palettized LM agrees with bf16's argmax on 92.1% of tokens against fp16's 94.6%
([lm-npu](lm-npu.md#precision)), and 4 bits (~490 MB) would free bandwidth.

- **What the export does** (measured 2026-11-21 on a layer used as `lm_step` uses it, `x @ W.T`):
  every trunk and head weight over 2^20 elements gets one k-means table of 2^N entries per 16 output
  channels, rows of the torch weight (tables [rows / 16, 1, 2^N, 1]); `convert_mlx.py` keeps the
  rows in order (the heads are concatenated, 2,048 rows each). A weight whose every 16 rows already
  hold at most 2^N values comes back exact: the export's k-means (scikit-learn here, kmeans1d where
  installed) finds them.
- **`fit`**: entries spread as the density's cube root over 1,024 equal-count bins, then 10 Lloyd
  iterations, an empty entry moved to its group's worst-fit value. On 4 random groups each of
  the update-16,000 student's `in_proj`, `linear_out`, `text_linear` and heads, its error is 0.94–1.02
  times the export's own k-means at 6 bits and 0.94–0.97 at 4: snapping alone exports no worse.
- **Training** (`--palettize-bits N`, `--palettize-refit` 100): the forward and backward run on
  the weights snapped to their tables; the gradients there update the full-precision weights
  (straight-through), restored before AdamW steps. Validation scores the palettized model. The
  tables are refit on the full-precision weights at the start and every `--palettize-refit` steps,
  which must divide `--save-every`, so a resume and the snap tool get the tables the run had.
  Saved checkpoints hold the full-precision weights.
- **Export**: `python -m training.student.palettize --weight CKPT --bits N --out SNAPPED` snaps a
  checkpoint onto its tables; then `convert_mlx.py` (bf16) and `export_lm_coreml.py --bits N` as
  before.
- **The run** (planned): from the final student, distilled from its own bf16 weights
  (`--teacher-weight` the same checkpoint, `--teacher-config` its config), `--schedule decay` from
  a low peak over a few thousand updates, at 6 and at 4 bits. Judged by the Precision table's
  agreement on the Neural Engine and the full-set chrF against the plain export and against
  snapping without training.
- Checked on the CPU (tiny modules, scratch script): at most 2^N values a group, snapping
  idempotent, the full-precision weights restored exactly after a step, the gradient equal to that
  of a model holding the snapped weights.

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

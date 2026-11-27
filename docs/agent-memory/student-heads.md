# Heads student

Read when touching `training/student/heads.py` or `restart.sh student-heads`, or training, scoring or
deciding on the phone student with parallel codebook heads. Back to [CONTEXT.md](../../CONTEXT.md) ·
[Design](#design) · [Codebooks](#codebooks) · [Speed](#speed) ([inference](#inference),
[training](#training)) · [Runs](#runs) · [Tests](#tests). The cut, training and
distillation it shares with the depformer student are in [phone-student](phone-student.md); training
against the Core ML palettization in [student-palettize](student-palettize.md).

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
- For: 16-codebook speech (Qwen3-ASR's Vietnamese WER through Mimi 8.9% against 15.7% at 8,
  English's 4.9% against 6.1%; [plan](../vi_en_parity_plan.md), stage 7.1) and 16 source codebooks
  to hear (gain unmeasured).
- Cost on the phone: 8 more heads (~34M parameters, under one main layer's ~63M) and 16 more
  embedding tables (~67M, one row read from each a frame): 1.279B parameters against 1.178B. The
  heads still sample in one pass.
- Untested: codebooks 9–15 are sampled without seeing 8..j-1 of their frame (MusicGen found fully
  parallel sampling worse). The teacher's depformer has read them, so on those codebooks the KL has
  a floor the student can't reach: it learns the teacher's distribution averaged over them.
- Voice (2026-11-26, user on the phone: "really weird, like a very old man. nothing like the input
  voice"): on 5 of the user's Vietnamese mic clips, WeSpeaker ResNet34 similarity to the input and
  harmonics-to-noise ratio over voiced frames only (`/Volumes/data/hibiki-zero-evals/mac/heads-voice/`
  `hnr.py`; evaluation's `hnr_db` counts every loud frame and reads lower:
  [voice-scores](voice-scores.md)) (input 11.0 dB): teacher 0.36 / 14.4 dB, depformer student `g2000_q6`
  0.32 / 12.2, heads student 0.00–0.15 / 4.2–6.6 (the phone's release, `pal6c4000_bf16`, plain
  `s16k_q6` and `s112k_pal6_bf16` on the Mac alike). Audio sampling doesn't help: argmax on
  codebooks 8–15 or 1–15, or 0.5 / top 25 on 1–15. So the heads lose the voice, not the
  quantization, the training length or the phone. Decoding only codebooks 0–7 (`drive8.py`): the
  teacher keeps 0.30 / 14.5 dB, the heads reach 0.10–0.11 / 6.7–8.1, so the loss is already in
  codebooks 1–7, not only in the parallel 8–15. Every gate scored text only, so none caught it.
  Files: `/Volumes/data/hibiki-zero-evals/mac/heads-voice/`.
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
runtime runs them as Core ML on the Neural Engine, with Mimi encoding on the GPU: on the A15 from
cool the frame holds 52–54 ms for 10 minutes ([lm-npu](lm-npu.md#speed)).
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
  `student_heads14_cb16/`): done 2026-11-21 18:42 UTC, 37.4 h with 12 validations. Validations (teacher-forced loss on
  the VI→EN / EN→VI caches, not comparable with the 8-codebook run's; chrF / BLEU / length ratio /
  end offset in s, every set's health pass unless marked):

  | Update (date) | TF loss | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|---|
  | 16,000 (11-20) | 3.701 / 3.494 | 25.28 | 28.40 / 2.77 / 1.59 / 3.0 | 22.15 / 5.08 / 1.05 / 3.1 | 28.49 / 4.31 / 1.50 / 3.8 | 28.49 / 5.64 / 1.43 / 3.1 | 33.17 / 6.96 / 1.19 / 3.5 | 23.29 / 6.91 / 1.20 / 1.4 | 31.29 / 10.20 / 1.06 / 2.4 |
  | 32,000 (11-20) | 3.493 / 3.257 | 27.44 | 29.55 / 3.96 / 1.47 / 2.7 | 25.33 / 6.88 / 1.14 / 2.4 | 29.78 / 4.66 / 1.61 / 4.8 | 31.30 / 6.49 / 1.47 / 3.0 (1 empty) | 34.93 / 8.80 / 1.23 / 3.5 | 23.81 / 6.03 / 1.27 / 0.8 | 33.26 / 11.21 / 1.14 / 2.4 |
  | 48,000 (11-20) | 3.395 / 3.157 | 28.84 | 27.69 / 3.64 / 1.55 / 3.2 | 29.99 / 9.90 / 1.10 / 2.0 (1 empty) | 29.31 / 4.80 / 1.59 / 4.6 | 32.77 / 8.17 / 1.42 / 2.9 | 35.98 / 9.27 / 1.22 / 3.3 | 26.33 / 8.72 / 1.25 / 1.2 | 35.38 / 13.79 / 1.10 / 2.3 |
  | 64,000 (11-20) | 3.320 / 3.082 | 30.05 | 29.92 / 4.42 / 1.50 / 2.6 (1 empty) | 30.18 / 10.51 / 1.08 / 2.2 (1 empty) | 30.22 / 5.63 / 1.63 / 5.1 (2 empty) | 27.47 / 5.77 / 1.63 / 3.8 (2 empty) | 32.17 / 9.21 / 1.16 / 3.8 | 26.54 / 9.78 / 1.18 / 0.9 (4 empty) | 36.17 / 14.68 / 1.09 / 2.1 |
  | 80,000 (11-20) | 3.272 / 3.033 | 30.47 | 30.14 / 3.78 / 1.63 / 2.8 | 30.79 / 10.11 / 1.20 / 2.2 | 30.40 / 5.50 / 1.66 / 4.7 | 33.69 / 7.69 / 1.71 / 3.5 | 35.90 / 10.44 / 1.28 / 4.1 | 27.85 / 9.19 / 1.35 / 1.4 | 36.71 / 14.97 / 1.13 / 2.5 |
  | 96,000 (11-21) | 3.223 / 2.988 | 31.43 | 30.32 / 5.56 / 1.42 / 2.3 | 32.53 / 12.45 / 1.10 / 1.5 (1 empty) | 29.17 / 5.50 / 1.62 / 5.0 (1 empty) | 31.47 / 8.04 / 1.50 / 2.9 (1 empty) | 35.73 / 11.07 / 1.16 / 3.4 | 28.94 / 10.19 / 1.27 / 0.9 | 38.60 / 16.40 / 1.08 / 2.1 |
  | 112,000 (11-21) | 3.182 / 2.950 | 33.27 | 33.52 / 6.24 / 1.39 / 2.0 | 33.03 / 13.24 / 1.05 / 1.6 | 32.61 / 6.93 / 1.52 / 4.1 | 37.34 / 10.79 / 1.33 / 2.6 | 38.46 / 12.85 / 1.12 / 2.6 | 30.12 / 13.02 / 1.16 / 1.0 (1 empty) | 38.39 / 17.47 / 1.04 / 2.0 |
  | 128,000 (11-21) | 3.143 / 2.922 | 35.29 | 37.08 / 8.14 / 1.33 / 1.9 | 33.51 / 13.47 / 1.06 / 1.6 (1 empty) | 34.44 / 7.75 / 1.53 / 4.2 | 40.04 / 13.34 / 1.25 / 2.5 | 41.95 / 14.81 / 1.16 / 3.1 | 31.10 / 12.72 / 1.18 / 0.8 | 40.84 / 19.11 / 1.09 / 2.1 |
  | 144,000 (11-21) | 3.113 / 2.899 | 36.00 | 37.94 / 8.08 / 1.35 / 1.8 | 34.07 / 14.24 / 1.03 / 1.6 | 35.10 / 7.67 / 1.58 / 4.7 | 39.49 / 13.41 / 1.30 / 2.4 (1 empty) | 41.53 / 15.43 / 1.15 / 3.0 | 32.92 / 14.20 / 1.18 / 0.8 | 40.06 / 19.63 / 1.04 / 2.2 |
  | 160,000 (11-21) | 3.094 / 2.889 | 36.59 | 38.29 / 8.48 / 1.35 / 1.9 | 34.90 / 14.84 / 1.05 / 1.7 | 34.97 / 8.31 / 1.55 / 4.4 | 38.45 / 11.75 / 1.32 / 2.6 | 41.75 / 15.47 / 1.13 / 2.7 | 32.77 / 14.36 / 1.17 / 0.8 | 40.71 / 19.98 / 1.04 / 1.9 |
  | 176,000 (11-21) | 3.087 / 2.881 | 36.99 | 38.34 / 8.64 / 1.33 / 1.7 | 35.64 / 15.57 / 1.05 / 1.6 | 35.10 / 8.16 / 1.58 / 4.6 (2 empty) | 39.60 / 12.99 / 1.31 / 2.6 | 41.23 / 16.20 / 1.10 / 2.8 | 31.01 / 12.62 / 1.15 / 0.8 | 40.94 / 20.01 / 1.06 / 2.1 |
  | 190,126, final (11-21) | 3.086 / 2.880 | 36.92 | 38.78 / 8.96 / 1.30 / 1.7 | 35.06 / 15.08 / 1.05 / 1.6 | 35.41 / 8.27 / 1.56 / 4.6 | 39.17 / 13.04 / 1.24 / 2.6 | 41.78 / 16.02 / 1.13 / 3.0 | 33.02 / 14.42 / 1.17 / 0.9 (1 empty) | 40.85 / 19.59 / 1.06 / 2.1 |

  At 16,000 against 8 codebooks: VI→EN +1.2 to +2.4 chrF on every set, EN→VI −1.0 (FLEURS) to +1.3,
  end offsets 0.1–0.7 s later; generation and scoring at 16 codebooks work on the GPU.
  At 64,000 (promoted): FLEURS dev rose both ways and `en-vi-val-en` (EN→VI) held, but `en-vi-val`
  (VI→EN) fell against 48,000: cv27 32.77 → 27.47 (length ratio 1.42 → 1.63), libriheavy 35.98 →
  32.17. Empty outputs: 10 over the sets, against 0–1 at the earlier validations.
  At 80,000 (promoted) `en-vi-val` recovered (cv27 33.69, libriheavy 35.90) with no empty outputs,
  but outputs grew longer on every set (length ratio 1.20–1.71). At 96,000 (promoted) they came
  back (1.08–1.62; VI→EN cv27 1.71 → 1.50), and EN→VI rose on all three sets.
  At 64,000 against the depformer student's own 64,000 (plain SFT, `student_dep10`): +2.9 to +8.2
  chrF on 6 of the 7 sets, −3.3 on `en-vi-val` cv27; FLEURS selecting 30.05 against 25.41, and
  30.53 for the depformer student after distillation (~196,000 updates), 35.50 after GRPO.
  At 96,000: FLEURS 30.32 / 32.53 (31.43) against `student_dep10`'s 28.67 / 24.53 (26.60).
  At 112,000 (promoted, +1.8 selecting, VI→EN +2.7 to +5.9 on every set) it passed the depformer
  student after distillation on FLEURS (33.52 / 33.03 against 31.74 / 29.31) and `en-vi-val-en`
  cv27 (30.12 against 28.76), and was 0.3–2.4 behind on the other four.
  At 128,000 (promoted, +2.0 selecting; up on every set, +0.5 to +3.6) the mean of
  the seven sets was 36.99, against 34.34 for the depformer student after distillation (its
  best, 64,000 of `student_dep10_anneal`), and the selecting mean 35.29 against 35.50 for that
  student after GRPO.
  At 144,000 (promoted, +0.7 selecting) FLEURS both ways, MultiMed-ST and `en-vi-val-en` cv27
  rose (+0.6 to +1.8), and `en-vi-val` and `en-vi-val-en` libriheavy fell 0.4–0.8. The seven-set
  mean was 37.30, and the selecting mean 36.00 passed the depformer student after GRPO (35.50).
  At 160,000 (promoted, +0.6 selecting: FLEURS +0.4 VI→EN, +0.8 EN→VI) the other five sets moved
  −1.0 (`en-vi-val` cv27) to +0.7; the seven-set mean was 37.41.
  At 176,000 (promoted, +0.4 selecting, nearly all of it FLEURS EN→VI +0.7) the seven-set mean
  held at 37.41: `en-vi-val` cv27 +1.2, `en-vi-val-en` cv27 −1.8, the rest within ±0.5.
  The final update (190,126, 2026-11-21 18:42 UTC) scored 36.92 selecting, so the run's promoted
  best stays 176,000 (`best_step176000`, hard-linked on the box as
  `weights/student_heads14_cb16_best.safetensors`); its seven-set mean was 37.72 (`en-vi-val-en`
  cv27 +2.0), so it starts the on-policy stage's run (user, 2026-11-22;
  [on-policy-distillation](on-policy-distillation.md#experiments)), on the box as
  `weights/student_heads14_cb16_final.safetensors`.

## Tests

`student_heads14_cb16`'s best (176,000, before any on-policy stage) on the test sets, as the
depformer student's shipped update 2,000 (after GRPO) was scored
([student-grpo](student-grpo.md#tests)): `/workspace/test_heads.sh`, seed 43, batch 64, text only,
LAAL (the box, 2026-11-21, 6 min; outputs in `runs/student_heads14_cb16/test_best176000/`).
OneVoice's clips aren't on the box: 11 sets. Every set passes health.

| Set | chrF | BLEU | EO (s) | LAAL (s) | Length | Empty | Update 2,000 chrF | EO (s) | LAAL (s) | Length |
|---|---|---|---|---|---|---|---|---|---|---|
| fleurs-vi/test | 36.78 | 8.21 | 1.7 | 2.8 | 1.32 | 0/857 | 35.65 | 0.88 | 2.32 | 1.25 |
| fleurs-en/test | 34.75 | 14.41 | 1.5 | 2.5 | 1.06 | 0/641 | 34.53 | 1.18 | 2.12 | 1.07 |
| multimed-vi/select | 34.99 | 8.17 | 4.6 | 3.2 | 1.56 | 0/1718 | 34.91 | 2.22 | 2.40 | 1.25 |
| multimed-vi/final | 34.94 | 8.31 | 4.6 | 3.2 | 1.55 | 1/1719 | 35.18 | 2.24 | 2.43 | 1.25 |
| giga-heldout/test | 61.64 | 28.82 | 3.9 | 2.4 | 1.84 | 0/1000 | 62.45 | 1.91 | 1.84 | 1.09 |
| en-vi-val/cv27 | 38.78 | 12.79 | 2.6 | 2.6 | 1.34 | 0/180 | 38.72 | 1.75 | 2.26 | 1.18 |
| en-vi-val/libriheavy | 42.32 | 16.06 | 3.1 | 3.1 | 1.16 | 0/180 | 42.91 | 1.91 | 2.59 | 1.04 |
| en-vi-val-noisy/cv27 | 39.37 | 11.92 | 2.7 | 2.6 | 1.33 | 0/180 | 36.21 | 1.98 | 2.42 | 1.20 |
| en-vi-val-noisy/libriheavy | 41.22 | 14.19 | 3.6 | 3.0 | 1.17 | 0/180 | 39.92 | 1.83 | 2.56 | 1.06 |
| en-vi-val-en/cv27 | 32.51 | 14.18 | 0.8 | 1.8 | 1.16 | 1/180 | 32.28 | 0.64 | 1.80 | 1.15 |
| en-vi-val-en/libriheavy | 41.27 | 20.29 | 2.0 | 2.4 | 1.05 | 0/180 | 41.50 | 1.61 | 2.20 | 1.06 |

- chrF: mean 39.87 over the 11 sets against update 2,000's 39.48; the noisy sets +1.3 and +3.2,
  `giga-heldout` −0.8, the rest within ±0.6.
- The final update (190,126, `test_final190126`) scored a mean 39.81: within ±0.9 of the best on
  every set but the noisy cv27 (−1.7), with the same lengths and latency: a tie.
- Latency and length: later on every VI→EN set (End Offset +0.8 to +2.4 s, LAAL +0.3 to +0.8 s)
  and longer (`giga-heldout` 1.84 against 1.09, BLEU 28.8 against 36.5; MultiMed-ST 1.56 against
  1.25). EN→VI is close. GRPO taught the depformer student to stop; the on-policy stage has the
  same job here.

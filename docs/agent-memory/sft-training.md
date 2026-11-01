# Full-model SFT training

Read when touching `training/h100.sh` or `training/finetune/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Trainer](#trainer) · [Objective](#objective) ·
[Launcher](#launcher) · [Checkpoints and recovery](#checkpoints-and-recovery) ·
[Side tools](#side-tools) · [Evaluation sets](#evaluation-sets)

The pod procedure is in [docs/finetune.md](../finetune.md). Caches are covered in
[sft-data-caches](sft-data-caches.md).

## Trainer

`training/finetune/train.py`:

- Full-model SFT from `kyutai/hibiki-zero-3b-pytorch-bf16`; every parameter is trainable.
- Inputs are one or more `--cache-dir` grounded caches plus a `--val-cache-dir`; rows over
  `--max-frames` (280) are dropped.
- It trains on streams (`training/finetune/streams.py`): consecutive cache rows joined into
  multi-sentence sequences of `--stream-seconds` (30–75 s, upstream's coarse-stage range).
  PhoMT rows of one source chunk (`phomt_s<N>r<M>`, which spans talk boundaries) stay in order;
  every other row is random, Libriheavy included: its segment numbers are not reading order and
  segments overlap ([en-speech-corpus](en-speech-corpus.md#quality-filters)). A stream closes when
  the next row overflows its drawn length, unless it is still under the minimum; the length range
  must exceed the longest row plus a long gap, so no stream but the last ends short. Each row's English starts its cached delay after its Vietnamese, never before the
  previous English ends; a 0.3–1.5 s gap (3–10 s one time in ten) separates the Vietnamese, and
  is stretched so the previous English overhangs the next Vietnamese by at most 2 s. Silent
  stretches hold slices of an encoded-silence track on both sides and PAD text, all supervised;
  the last row keeps the source and text EOS, and the source after its EOS stays silence. The
  cache itself is never changed. The silence track is encoded on the CPU in float32, so every
  machine and resume gets the same codes but for a rare residual near-tie (1 of 1,876 frames
  between thread counts).
- Mimi codes depend on the encoder's state. Inference encodes one stream from a fresh state, so
  a stream opens with the encoder's start-up codes (the silence track's first 16 frames) and is
  steady silence elsewhere. Every cached clip was encoded alone from a fresh state, so unless a
  piece opens the stream its Vietnamese frame 0 becomes silence (a start-up marker: codebook 0 is
  1049, 1698 or 995 in ~90% of rows; frame 1 already has ~200 values), and so does English silence
  before the first word within a clip's first 16 frames. What stays different from a continuous
  encoding: Vietnamese frame 0 sometimes holds a syllable onset (above −20 dB of the clip peak in
  28% of 400 rows), 23% of rows have English speech inside their first 16 frames, where codebook 0
  matches a continuous encoding 60–80% of the time, and residual codebooks differ near every join
  (on 40 CV clips against a 2.56 s silence pre-roll, ~42% of all codes match even beyond 2 s:
  Mimi's 10 s context). Encoding each clip after a silence pre-roll would be the full fix (a
  Mimi-only re-encode, about half the cache build).
- The plan is drawn once with `--seed` (42); streams are length-sorted and grouped into batches
  of at most `--batch-frames` (4,608, as 16 rows of 288 frames before) padded frames, whose
  order is shuffled once; every epoch replays it, and resume skips the planned batches.
  Validation streams and their batch order use the fixed `VALIDATION_SEED`, so `validate.py`
  scores the same streams and `--val-batches` takes a random subset.
- Measured on 6,152 rows of `cv27-00010`, `libriheavy-medium-00050`,
  `libriheavy-large-00000-00003` and a PhoMT shard (2026-11-01): 1,421 streams of 47 s median
  (p10 33, p90 66; none but the last under 30 s), 4 rows each, 20% of the Vietnamese side
  silent; the 2 s overhang cap binds at the median join, since the English runs longer than the
  Vietnamese; batches hold 7 streams median at 97.6% fill.
- Defaults:
  - 1 epoch (`--max-steps` overrides);
  - peak LR 1e-6, linear warmup over 5% of the steps (`--warmup-fraction`), cosine decay to 0;
    fused AdamW with betas (0.9, 0.95) and weight decay 0.1; grad clip 1.0;
  - validation and checkpoints every 9,000 steps; `--val-every` must be a multiple of
    `--save-every`, because promotion hard-links the checkpoint saved at that step.
- Precision: fp32 master weights, bf16 autocast, TF32 matmul "high", and a causal-SDPA patch
  (streams stay under the 3,000-frame context, so the mask is purely causal). Batches pad to
  16-frame buckets (`streams.FRAME_BUCKET`), and training requires moshi's internal compile,
  meaning `NO_TORCH_COMPILE` must be unset. `training.finetune` is CUDA-only (`train.py` has no
  device flag); the `training.prep` steps, the cache build included
  ([sft-data-caches](sft-data-caches.md)), run on the Mac.

## Objective

`common.compute_batch_losses`, Kyutai's `moshi-finetune` weighting (decided 2026-11-01,
[plan](../vi_en_parity_plan.md#decisions-2026-11-01)):

- Audio: the weighted mean CE over the 16 English target codebooks, codebook 0 (semantic)
  weighted `--first-codebook-weight` 100 against 1 for each acoustic codebook.
- Text: the weighted mean CE over every text position, a PAD weighted `--text-pad-weight` 0.5
  against 1 for a word token or EOS.
- Loss = 1.0 × audio + 1.0 × text. Logs and validation also report the semantic and acoustic
  CE and the word and PAD CE and accuracy unweighted.
- On the upstream model and two 96-frame streams (Mac CPU, fp32): loss 8.61, word CE 13.3 (it
  knows no Vietnamese), PAD accuracy 0.64.
- The previous objective (uniform codebooks; only PADs before the first word supervised, at
  0.05) is described in [docs/analysis/loss_function.md](../analysis/loss_function.md) for the
  run it trained.

## Launcher

`training/h100.sh` runs from the repo root and invokes modules as
`python -m training.<package>.<module>`.

- `setup` needs Linux, driver 570+, exactly one H100, Python 3.10–3.13 and no existing
  `.venv`. It installs Torch `2.8.0+cu128` (CUDA 12.8), pinned dependencies, and
  `moshi==0.2.13 --no-deps`.
- `preflight` checks:
  - a clean tracked tree;
  - exact Torch and Moshi versions; one H100 with compute capability 9.0;
  - ≥90 GiB VRAM (a standard 80 GB H100 fails), ≥110 GiB host RAM and ≥190 GiB free disk;
  - the SHA-256 of the four upstream files in `weights/`.
- `train RUN_NAME ARGS…` runs preflight and requires an empty `runs/RUN_NAME/`. It needs a
  public `HIBIKI_HF_REPO` whose `RUN_NAME/` prefix is empty. It writes `run_id.json` (commit
  plus random id), uploads it as `RUN_NAME/run.json`, and runs `train.py --out-dir
  runs/RUN_NAME ARGS…` beside the `hf_sync` watcher. Training is killed if sync dies.
- `resume runs/RUN_NAME/trainer_stepNNNNNN.pt ARGS…` requires:
  - a checkpoint directly under `runs/`;
  - a remote no newer than local;
  - a `run_id.json` commit equal to HEAD.
  It syncs once, then resumes with the same ARGS.

## Checkpoints and recovery

- A complete pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt`; two are kept
  locally. `best_step*.safetensors` is a hard link promoted only on a strict improvement in
  teacher-forced validation loss (`best.json`, metric `teacher_forced_loss`).
- `train.py` resumes only from the newest complete pair, and only if `run_config.json`
  matches on `RESUME_KEYS`: data dirs, weights, hyperparameters, cadences, seed, train row
  count and total steps. From the first validation step on, `best.json` is required.
- `hf_sync.py` uploads under `HIBIKI_HF_PREFIX` (the run name):
  - `checkpoints/`: on each poll, the newest complete pair if newer than the remote, keeping
    the newest two;
  - `best/`: the newest promoted model only;
  - `metadata/run_config.json`;
  - logs, on the final sync only.
  It requires a public repo and a remote `run.json` byte-equal to the local `run_id.json`. It
  polls every 60 s and aborts after three failures.
- `hf_sync` never restores anything; restoring is the manual procedure in
  [docs/finetune.md](../finetune.md).

## Side tools

- `eval.py` is the correct-source free-running check on a pair file (`id`, `vi_audio`,
  `text_en`). Despite its name, `run_greedy_eval` samples (text 0.4, audio 0.8).
- `validate.py` reports teacher-forced losses with BF16 weights on the validation streams of one
  cache, planned as `train.py` plans them.
- `common.py` is shared by `train.py`, `validate.py` and `eval.py`.
- `train.py` and `eval.py` take a required `--out-dir`.

## Evaluation sets

`training.prep.eval_sets` (huggingface_hub and pyarrow only, so it runs on the Mac or the pod)
writes `eval.py` pair files of real Vietnamese speech under `data/eval/`; they are never trained
on ([plan](../vi_en_parity_plan.md#2-evaluation-sets)). Built and checked on 2026-11-01:

| File | Rows | Hours | Use |
|---|---:|---:|---|
| `fleurs-vi/dev.jsonl` | 361 | 1.19 | checkpoint selection |
| `fleurs-vi/test.jsonl` | 857 | 3.00 | final report |
| `multimed-vi/select.jsonl` | 1,718 | 3.01 | non-regression gate |
| `multimed-vi/final.jsonl` | 1,719 | 3.02 | final report |

- FLEURS `google/fleurs` at `70bb2e8`: `vi_vn` 16 kHz WAV read speech; `text_en` is the `en_us`
  raw text of the same sentence id (every Vietnamese row has one). Several speakers read each
  sentence, so ids are the WAV stems.
- MultiMed-ST `leduckhai/MultiMed-ST` at `d8c3b30`, Vietnamese `corrected.test` (3,437 rows),
  split by a `random.Random(0)` shuffle into `select` and `final`. Every clip is **8 kHz** mono
  Vorbis (the card's 16 kHz is the decode rate): telephone band. Segments are cut mid-phrase
  (one starts with the tail of "bệnh viện") while the English is a clean sentence, so read chrF
  rather than BLEU. Only the test split's English was corrected by annotators; `eval` and
  `train` are raw Gemini drafts and are not used. No conversation ids exist, so the halves may
  share conversations.

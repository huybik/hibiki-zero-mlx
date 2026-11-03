# Full-model SFT training

Read when touching `training/h100.sh` or `training/finetune/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Trainer](#trainer) · [Speed](#speed) ·
[Objective](#objective) · [Launcher](#launcher) ·
[Checkpoints and recovery](#checkpoints-and-recovery) · [Side tools](#side-tools) ·
[Evaluation sets](#evaluation-sets)

The pod procedure is in [docs/finetune.md](../finetune.md). Caches are covered in
[sft-data-caches](sft-data-caches.md).

## Trainer

`training/finetune/train.py`:

- Full-model SFT from `kyutai/hibiki-zero-3b-pytorch-bf16`; every parameter is trainable.
- Inputs are one or more `--cache-dir` grounded caches plus a `--val-cache-dir`; rows over
  `--max-frames` (280) are dropped.
- It trains on streams (`training/finetune/streams.py`): consecutive cache rows joined into
  multi-sentence sequences of `--stream-seconds` (30–120 s, `streams.STREAM_SECONDS`, decided
  2026-11-02: upstream's coarse stage used 30–75 s and its RL stage 120 s, and upstream degrades
  after ~1.6 minutes of input, so streams practise long input).
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
  wherever nobody speaks a stream holds the silence track at the same frames: digital silence
  encoded from a fresh state, start-up codes in its first 16 frames, then steady silence. Every cached clip was encoded alone from a fresh state, so unless a
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
  `libriheavy-large-00000-00003` and a PhoMT shard (2026-11-02, 30–120 s): 981 streams of 68 s
  median (p10 37, p90 104; none under 30 s), 6 rows each, 21% of the Vietnamese side silent; the
  2 s overhang cap binds at the median join, since the English runs longer than the Vietnamese;
  batches hold 4 streams median, streams filling 98.9% of the padded frames (88.5% of the
  4,608-frame budget; 30–75 s streams filled 92.4%).
- The full `en_speech` cache (3,441,391 rows) plans into 622,220 streams in 131,553 batches,
  one epoch (planned and every stream assembled on the pod, 2026-11-03).
- Rows load from per-shard packs (`streams.pack_shard`: one flat int32 `codes.bin` plus an
  index, in `<cache-dir>/.packed/<shard>/`, rebuilt when a shard's size or mtime changes),
  memory-mapped with `torch.from_file`, which keeps no file descriptor (numpy's memmap holds one
  per shard, past the 1,024 limit). Packing all 1,832 shards takes ~30 s on 32 forked workers;
  a start then loads the rows in 9 s with 3 GiB resident, against 455 s and ~75 GiB unpickling
  the shards. `--num-workers` (4) assemble streams in the background.
- Defaults:
  - 1 epoch (`--max-steps` overrides);
  - peak LR 1e-6, linear warmup over 5% of the steps (`--warmup-fraction`), cosine decay to 0;
    fused AdamW with betas (0.9, 0.95) and weight decay 0.1; grad clip 1.0;
  - checkpoints every 1,000 steps (~6.5 min) and validation every 9,000; `--val-every` must be
    a multiple of `--save-every`, because promotion hard-links the checkpoint saved at that step;
  - each validation translates `--eval-pairs` (FLEURS dev: 361 rows, 72 minutes of speech)
    free-running on a bf16 copy of the model, shortest first in batches of `--eval-batch-size`
    64, each batch generated until every row ends or 20 s past its longest row: at most 2,647
    generation steps. Mimi codes 8 rows at a time (`hibiki_helpers.MIMI_CHUNK_ROWS`).
- Precision: fp32 master weights, bf16 autocast, TF32 matmul "high". Batches pad to 16-frame
  buckets (`streams.FRAME_BUCKET`), and training requires torch compile (moshi's and
  `lm_training`'s), meaning `NO_TORCH_COMPILE` must be unset. `training.finetune` is CUDA-only
  (`train.py` has no device flag); the `training.prep` steps, the cache build included
  ([sft-data-caches](sft-data-caches.md)), run on the Mac.

## Speed

The fast path (`lm_training`), its measurements and the speed by GPU are in
[sft-speed](sft-speed.md).

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

- `setup` needs Linux, driver 580+ (CUDA 13), exactly one H100 or H200 (Hopper, compute
  capability 9.0: the kernels and measurements are sm_90's), Python 3.10–3.13 and no existing
  `.venv`. It installs Torch `2.15.0+cu134` (CUDA 13.4, the newest build on 2026-11-03), the
  newest of every pinned dependency (`huggingface-hub` 2.4), and `moshi==0.2.13 --no-deps`
  (moshi declares `torch<2.10`; 0.2.13 is its newest release and runs on 2.15).
- `preflight` checks:
  - a clean tracked tree;
  - exact Torch and Moshi versions; one H100 or H200 with compute capability 9.0;
  - ≥78 GiB VRAM (an 80 GB card shows 79.2), ≥80 GiB of RAM (the container's cgroup limit
    when lower than `/proc/meminfo`, which shows the host on Vast) and 130 GiB of disk: free
    space plus, for `run`, what the run dir already holds (each hard-linked inode once);
  - the SHA-256 of the four upstream files in `weights/`.
- `run RUN_NAME [ARGS…]` is the only way to train, and the same command starts and continues.
  It holds `flock` on `runs/.RUN_NAME.lock` (inherited by the trainer and the sync), so a
  second launcher of a running run exits; then preflight:
  - without `runs/RUN_NAME/run_id.json`: an empty run dir (a start cut short before
    `run_id.json` leaves only `train_args.json`, which is removed) and an empty `RUN_NAME/`
    prefix in a public `HIBIKI_HF_REPO`; it writes `train_args.json` (ARGS), then `run_id.json`
    (commit plus random id), then uploads it as `RUN_NAME/run.json`;
  - with it: the saved ARGS (other ARGS are refused), a remote pair no newer than the newest local
    one, a local `best.json` no older than the remote's newest `best/` marker, and a `run_id.json`
    commit equal to HEAD; a missing remote `run.json` (its upload failed) is uploaded if the prefix
    is otherwise empty.
  Either way it runs `train.py --out-dir runs/RUN_NAME ARGS…` beside the `hf_sync` watcher, and
  kills training if sync dies. INT or TERM to the launcher stops training and waits for the sync's
  final upload (`wait` is retried: a trapped signal interrupts it). On an interruptible pod the
  boot script runs it ([docs/finetune.md](../finetune.md#interruptible-pods)).

## Checkpoints and recovery

- A complete pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt` (12.5 + 25 GB);
  two are kept locally. `train.CheckpointWriter` copies the model and optimizer state into
  pinned host buffers on the training stream and returns; a thread waits for the copy, prunes
  to `--keep-checkpoints` − 1 pairs, writes the model then the trainer file (each fsynced before
  its rename, then the directory), and prunes again. The next save, a promotion (which
  hard-links the model file) and the end of training wait for it; a failed write raises at the
  next log step.
- Checkpoints are chosen on free-running translation of real Vietnamese (decided 2026-11-02:
  the last run's teacher-forced loss kept improving while its translations got worse). Each
  validation logs the teacher-forced losses of the cache's validation streams, then translates
  `--eval-pairs` (default `data/eval/fleurs-vi/dev.jsonl`, so build the
  [evaluation sets](#evaluation-sets) before training; rows need `duration_s`) with `eval.py`'s
  sampling, reseeded per batch from `--seed`. `best_step*.safetensors` is a hard link to the
  checkpoint that passes the
  generation health gates (`common.generation_health`) with a strictly higher corpus chrF
  (`best.json`: `step`, `model`, `metric` `free_running_chrf`, `chrf`); a run whose checkpoints
  all fail the gates promotes none. `val_log.jsonl` holds `teacher_forced` and `free_running`
  metrics per step, and `eval/stepNNNNNN/` the predictions.
- Training's chrF ranks the checkpoints of one run. It generates with bf16 weights, as `eval.py`
  does, but in length-sorted batches of 64, while `eval.py` defaults to batch 1 in file order, so
  the two give different numbers for the same checkpoint; compare runs with `eval.py`.
- A start in an out-dir with complete pairs resumes the newest one, and only if
  `run_config.json` matches on `RESUME_KEYS`: data dirs, weights, hyperparameters, cadences,
  seed, train row count, evaluation pairs and rows, and total steps. Lines of `train_log.jsonl`
  and `val_log.jsonl` past the resumed step are dropped. A start without a pair begins at step 0
  and removes the logs, `best.json` and `eval/` of a run cut short before its first save.
- A pair is saved before its validation, and a validation's log line is written after its promotion.
  A resume at a validation step (a multiple of `--val-every`) that `val_log.jsonl` lacks validates
  it first; a finished run exits 0 after loading the data and before loading the model (the boot
  hook's retry loop stops on it). Promotion relinks `best_stepS.safetensors` (a link to a temp name,
  then a rename) unless it is already this run's `model_stepS` (a replacement pod can restore best S
  beside an older pair, then retrain S). A torn last log line, or one missing its newline, is
  rewritten on resume. The writer's thread is not a daemon, so a Python exception still finishes the
  pair being written; the launcher's TERM kills it, and the run resumes from the pair before. A
  replacement pod has `val_log.jsonl` only if it restored the run's `artifacts/` (synced whenever
  the training process exits); without it, the resumed step's validation runs again and may promote
  it on sampling noise.
- `h100.sh run` refuses to continue unless the local `best.json` is at least as new as the
  recovery repo's newest `best/` marker: a resume without it would promote a worse checkpoint,
  and `hf_sync`, which keeps only the newest remote best, would replace the better one.
- Checked on the pods (2026-11-03), killing `train.py` with `kill -9` and rerunning the same
  command: on the H200 NVL, killed at step 70 with the step-60 pair half-written, it skipped
  that pair, resumed from step 40 and logged the same losses for steps 50–70; on the H100 SXM,
  killed once the step-80 pair was saved but before its validation was logged, it validated
  step 80 first, then trained to 160 at the s/step it had before the kill.
- Checked on the Mac CPU with a tiny random LM (2026-11-02): training, saving, both
  validations, promotion (a lower chrF keeps the promoted checkpoint, a higher one replaces it,
  a run failing the gates promotes none), resumes that rerun a validation cut short mid-run and
  at the final step with the original numbers. The launcher's best-checkpoint check was run on
  fake states.
- `hf_sync.py` uploads under `HIBIKI_HF_PREFIX` (the run name):
  - `checkpoints/`: on each poll, the newest complete pair at a validation step
    (`run_config.json`'s `val_every`) if newer than the remote, keeping the newest two; the
    final sync, after the training process exits, uploads the newest pair whatever its step;
  - `best/`: the newest promoted model only;
  - `metadata/run_config.json` and `metadata/train_args.json`;
  - logs, on the final sync only.
  It requires a public repo and a remote `run.json` byte-equal to the local `run_id.json`. It
  polls every 60 s and aborts after three failures.
- `hf_sync` never restores anything; restoring is the manual procedure in
  [docs/finetune.md](../finetune.md).

## Side tools

- `eval.py` is the correct-source free-running check on a pair file (`id`, `vi_audio`,
  `text_en`). Its defaults are `common.GenerationConfig`'s, which training's checkpoint
  selection uses too.
  - Despite its name, `run_greedy_eval` samples every token at temperature 0.8, top-k 250, as
    the paper does (Sec. 4.5).
  - It generates until every row has emitted the text EOS, or 20 s past the batch's longest
    source.
  - Each prediction records `end_offset_s`, the time from the end of the source audio to the
    last generated word; the metrics report their median.
  - Decided 2026-11-03. At text temperature 0.4 the step-18,000 model fell behind, stalled and
    ended 11 of 48 FLEURS rows, against 27 at 0.8 with the same 8 s tail and 33 with a longer
    one. Its EOS came up to 14 s after a row ended, and the 8 s tail cut a third of them
    ([analysis](../analysis/eos_and_lag.md)).
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

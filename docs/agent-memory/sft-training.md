# Full-model SFT training

Read when touching `training/finetune/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Trainer](#trainer) · [Speed](#speed) ·
[Objective](#objective)

The pod procedure is in [docs/finetune.md](../finetune.md). Caches are covered in
[sft-data-caches](sft-data-caches.md). The launcher, checkpoints and recovery are in
[sft-recovery](sft-recovery.md); checkpoint selection, the side tools and the evaluation sets in
[sft-evaluation](sft-evaluation.md).

## Trainer

`training/finetune/train.py`:

- Full-model SFT from `kyutai/hibiki-zero-3b-pytorch-bf16`; every parameter is trainable.
- Inputs are one or more `--cache-dir` grounded caches plus a `--val-cache-dir`; rows over
  `--max-frames` (280) are dropped.
- It trains on streams (`training/finetune/streams.py`): consecutive cache rows joined into
  multi-sentence sequences of `--stream-seconds` (30–120 s, `streams.STREAM_SECONDS`, decided
  2026-11-02: upstream's coarse stage used 30–75 s and its RL stage 120 s, and upstream degrades
  after ~1.6 minutes of input, so streams practise long input).
  The rows of one document stay together in order: a tranche row's `doc` and `seq` (a
  PhoAudiobook reader's run, a GigaSpeech 2 recording; a viVoice row is its own doc,
  [tranche-cache](tranche-cache.md)) and a PhoMT source chunk (`phomt_s<N>r<M>`, which
  spans talk boundaries, its doc and seq read from the id); every other row is random, Libriheavy included: its segment numbers are not reading order and
  segments overlap ([en-speech-corpus](en-speech-corpus.md#quality-filters)). A stream closes when
  the next row overflows its drawn length, unless it is still under the minimum; the length range
  must exceed the longest row plus a long gap, so no stream but the last ends short. Each row's English starts its cached delay after its Vietnamese, never before the
  previous English ends; a 0.3–1.5 s gap (3–10 s one time in ten) separates the Vietnamese, and
  is stretched so the previous English overhangs the next Vietnamese by at most 2 s. Silent
  stretches hold slices of an encoded-silence track on both sides and PAD text, all supervised;
  the last row keeps the source and text EOS, and the source after its EOS stays silence. A tenth
  of the streams (`streams.SINGLE_ROW_STREAMS`, from 2026-11-04) hold one row, lead to both EOS,
  as a single-sentence evaluation feeds it: the first run, on 30–120 s streams only, never saw the
  source end after one sentence and never ended a third of the FLEURS rows. On the three shards
  below (seed 43) they are 9.8% of the streams and 2.4% of the frames (median 200 frames). The
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
- Multi-row streams run 68 s median (p10 37, p90 104), 6 rows each, with 21% of the Vietnamese
  side silent; the 2 s overhang cap binds at the median join. Batches hold 4 streams median,
  filling 98.9% of the padded frames and 88.5% of the 4,608-frame budget (6,152 rows of four
  shards, 2026-11-02, before single-row streams).
- The full `en_speech` cache (3,441,391 rows) plans one epoch of 131,703 batches with seed 43
  (the second run). Single-row streams changed every plan, the validation streams' too, so
  teacher-forced validation losses do not compare across that change (2026-11-04).
- Rows load from per-shard packs (`streams.pack_shard`: one flat int32 `codes.bin` plus an
  index and each row's doc and seq, in `<cache-dir>/.packed/<shard>/`, rebuilt when a shard's
  size or mtime changes or `PACK_VERSION` does: 3 since the docs, 2026-11-05),
  memory-mapped with `torch.from_file`, which keeps no file descriptor (numpy's memmap holds one
  per shard, past the 1,024 limit). Packing all 1,832 shards takes ~30 s on 32 forked workers;
  a start then loads the rows in 9 s with 3 GiB resident, against 455 s and ~75 GiB unpickling
  the shards. `--num-workers` (4) assemble streams in the background.
- Defaults:
  - 1 epoch (`--max-steps` overrides);
  - peak LR 1e-6, linear warmup over 5% of the steps (`--warmup-fraction`), cosine decay to 0;
    fused AdamW with betas (0.9, 0.95) and weight decay 0.1; grad clip 1.0. Each train-log line
    holds its steps' mean and max gradient norm before clipping (`grad_norm`, `max_grad_norm`),
    summed on the GPU without a sync per step. A non-finite norm raises at the next log step and
    before any save: the losses cannot show the last step's update, and a NaN pair saved would be
    resumed by every restart;
  - checkpoints every 1,000 steps (~6.5 min) and validation every 9,000; `--val-every` must be
    a multiple of `--save-every`, because promotion hard-links the checkpoint saved at that step;
  - each validation translates the rows of every `--eval-pairs` file (FLEURS dev, 361 rows, 72
    minutes of speech, and `en-vi-val` Common Voice and Libriheavy, 180 rows each, 54 minutes)
    in one pass, free-running on a bf16 copy of the model (built without reading
    `--model-weight`), shortest first in batches of
    `--eval-batch-size` 64 that mix the sets, each batch generated until every row ends or 20 s
    past its longest row: at most 4,918 generation steps. Mimi codes 8 rows at a time
    (`hibiki_helpers.MIMI_CHUNK_ROWS`).
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
- The upstream model on two 96-frame streams: loss 8.61, word CE 13.3, PAD accuracy 0.64. The
  previous objective is in [docs/analysis/loss_function.md](../analysis/loss_function.md).

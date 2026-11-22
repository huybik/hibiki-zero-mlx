# Full-model SFT training

Read when touching `training/finetune/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Trainer](#trainer) · [Target timing](#target-timing) ·
[Schedules and averaging](#schedules-and-averaging) · [Speed](#speed) · [Objective](#objective)

The pod procedure is in [docs/finetune.md](../finetune.md). Caches are covered in
[sft-data-caches](sft-data-caches.md). The launcher, checkpoints and recovery are in
[sft-recovery](sft-recovery.md); checkpoint selection, the side tools and the evaluation sets in
[sft-evaluation](sft-evaluation.md).

## Trainer

`training/finetune/train.py`:

- Full-model SFT from `kyutai/hibiki-zero-3b-pytorch-bf16`; every parameter is trainable.
  `common.load_policy` builds the model for SFT and GRPO alike, through `common.build_lm` (moshi's
  `get_moshi`, or for a parallel-heads student its own load, [student-heads](student-heads.md#design)),
  which, loading a Hibiki weight, copies the text PAD's input embedding over the text EOS's (its
  way to make a sampled early EOS read on as PAD). So every run starts with that row reset, and
  it trains from there: cosine 0.945 to PAD's after round 3, 0.88 after round 4. `eval.py`, the
  trainers' validation and a resumed pair keep a checkpoint's own row; a tool that loads a
  checkpoint through `get_moshi` resets it. Only the frames after a sampled EOS can differ, and
  evaluation cuts a row's text at its first EOS (found 2026-11-10).
- Inputs are one or more `--cache-dir` grounded caches plus one or more `--val-cache-dir`
  (each direction's, each the others' pair; each is planned and scored apart), and every row's
  pair in the other direction, from those or from `--pair-cache-dir` ([Target
  timing](#target-timing)); rows over `--max-frames` (280) with their target where the plan puts
  it are dropped. `--tokenizer` defaults to the start weight's, and every
  shard must be on it, except that a run on the Vietnamese swap reads base-tokenizer shards
  without the rows the swap reassigned ([two-way](two-way.md#tokenizer-checks)).
- It trains on streams (`training/finetune/streams.py`): consecutive cache rows joined into
  multi-sentence sequences of `--stream-seconds` (30–120 s, `streams.STREAM_SECONDS`, decided
  2026-11-02: upstream's coarse stage used 30–75 s and its RL stage 120 s, and upstream degrades
  after ~1.6 minutes of input, so streams practise long input). A model with fewer codebooks a
  stream than the cache (a phone student) gets each assembled stream cut to its rows
  (`streams.model_rows`, [phone-student](phone-student.md#training-a-student)).
  The rows of one document stay together in order: a tranche row's `doc` and `seq` (a
  PhoAudiobook reader's run, a GigaSpeech 2 recording; a viVoice row is its own doc,
  [tranche-cache](tranche-cache.md)) and a PhoMT source chunk (`phomt_s<N>r<M>`, which
  spans talk boundaries, its doc and seq read from the id); every other row is random, Libriheavy included: its segment numbers are not reading order and
  segments overlap ([en-speech-corpus](en-speech-corpus.md#quality-filters)). A stream closes when
  the next row overflows its drawn length, unless it is still under the minimum; the length range
  must exceed the longest row plus a long gap, so no stream but the last ends short. A stream
  holds one direction's rows: each direction is planned apart ([two-way](two-way.md)). Each row's
  target starts its cached delay after its source, or later where the floor puts it ([Target
  timing](#target-timing)), never before the previous target ends; a
  0.3–1.5 s gap (3–10 s one time in ten) separates the sources, and is stretched so the previous
  target overhangs the next source by at most 2 s. Silent
  stretches hold slices of an encoded-silence track on both sides and PAD text, all supervised;
  the last row keeps the source and text EOS, and the source after its EOS stays silence. A tenth
  of the streams (`streams.SINGLE_ROW_STREAMS`, from 2026-11-04) hold one row, lead to both EOS,
  as a single-sentence evaluation feeds it: the first run, on 30–120 s streams only, never saw the
  source end after one sentence and never ended a third of the FLEURS rows.
- Late starts: 5% of rows, in any stream (`streams.LATE_START`), start their target at a frame
  drawn from [half the source, 1 s past its end] instead of their start ([Target
  timing](#target-timing)) when that is later; one that would push its stream past the longest
  keeps its start. The cache draws every delay from U[0, 0.5 ×
  the source's duration] (`cache_rows.target_delay_s`), so a first word rarely comes after its
  source has ended: measured on the box (2026-11-11, 12 packs a cache), 0.04% of the GigaSpeech 2
  and `en-vi` validation rows, 0.87% of the `en_speech` train rows and 1.7% of the `vi-en`
  validation rows (2.6% of its CV27 rows, 0% of its Libriheavy ones: the English Common Voice
  clip's own leading silence). Planned into streams, where a target also waits for the previous
  one, 8% of `vi-en` validation rows had their first word after the source, 0.05% of `en-vi`
  ones. A model whose start
  slipped past a short clip's end then stayed silent through the evaluation's 20 s tail: the
  two-way run at step 36,000 left 16 of FLEURS-vi dev's 361 rows and 48 of FLEURS-en dev's 392
  empty, nearly all under 9 s (its start model 3 FLEURS-vi rows). Drawn over the second half of
  the source and past it, late starts leave no stretch of a sentence where a start has never been
  seen. On the validation caches' rows (both directions, the SFT plan): 21% (`en-vi`) and 33%
  (`vi-en`) of late targets start after their source, and with the target's own lead 23% and 57%
  of their first words come after it, ~85% of those in a gap without a source EOS, as
  continuous speech gives; rows with a first word after their source go from 0.05% to 1.2%
  (`en-vi`) and 8.0% to 10.2% (`vi-en`), while the median onset moves ~0.1 s (`en-vi` 2.00 →
  2.08 s, `vi-en` 2.80 → 2.88 s). Starts slipping more than ~1.5 s past an `en-vi` source stay
  unseen: if short clips still go silent, widen `LATE_START_FRAMES`. Late starts change every
  stream plan, the teacher-forced validation streams included, so validation losses do not
  compare across them.
- The MLX and Swift runtimes end a file's input as the evaluation does (the source EOS, then up
  to 20 s of silence, stopping at the text EOS), so a start in the silence after a short clip has
  the same room there as in training's validation
  ([mlx-inference](mlx-inference.md#runtime)).
- The cache itself is never changed. The silence track is encoded on the CPU in float32, so every
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
  order is shuffled once; every epoch replays it, and resume skips the planned batches. Each
  batch is one update, computed whole: 4,608 frames fit an 80 GB card, round 4's 8,192 need ~97
  GiB at once. Round 4 ran them on the H200; the two-way run on the H100 in slices of its streams
  (`--micro-frames`, 6.5% slower, removed with the move to 4,608: [sft-speed](sft-speed.md)).
  Whole 4,096-frame batches accumulated two to an update ran out of memory there: the first's
  12 GB of fp32 gradients stay while the second's activations build.
  Validation streams and their batch order use the fixed `VALIDATION_SEED`, so `validate.py`
  scores the same streams and `--val-batches` takes a random subset of each cache's.
- Multi-row streams run 68 s median (p10 37, p90 104), 6 rows each, with 21% of the Vietnamese
  side silent; the 2 s overhang cap binds at the median join. Batches hold 4 streams median,
  filling 98.9% of the padded frames and 88.5% of the 4,608-frame budget (6,152 rows of four
  shards, 2026-11-02, before single-row streams).
- The full `en_speech` cache (3,441,391 rows) plans one epoch of 131,703 batches with seed 43
  (the second run). Single-row streams changed every plan, the validation streams' too, so
  teacher-forced validation losses do not compare across that change (2026-11-04).
- Rows load from per-shard packs (`streams.pack_shard`: one flat int32 `codes.bin` plus an
  index and each row's doc and seq, in `<cache-dir>/.packed/<shard>/`, rebuilt when a shard's
  size or mtime changes or `PACK_VERSION` does: 5 since each pack records its shard's direction
  and tokenizer and the rows' text tracks in `text.bin`, 2026-11-10; [two-way](two-way.md)),
  memory-mapped with `torch.from_file`, which keeps no file descriptor (numpy's memmap holds one
  per shard, past the 1,024 limit). Packing all 1,832 shards takes ~30 s on 32 forked workers;
  a start then loads the rows in 9 s with 3 GiB resident, against 455 s and ~75 GiB unpickling
  the shards. The trainer reads only packs: `python -m training.finetune.streams --drop-shards
  DIR...` packs and then deletes the shards (only once the Hub lists them), marking each pack
  `dropped`. A pack so marked loads without its shard if it is of the current `PACK_VERSION` (a
  new version needs the shard fetched again); one left without its shard any other way raises, so
  deleting a shard never leaves its rows in a run unnoticed. A dropped shard keeps its place in the
  name order, so the plan does not change. `--num-workers` (4) assemble streams in the background.
- Defaults:
  - 1 epoch (`--max-steps` overrides);
  - 4,608-frame batches at peak LR 2.25e-6 (user, 2026-11-13): round 4's 3e-6 at 8,192 frames
    scaled by sqrt(4,608 / 8,192). With AdamW each update moves the weights about the rate
    whatever the batch, so 4,608 at 3e-6 moved them 1.78 times as far per hour of data as round 4,
    and this model loses real speech when pushed: the learning-rate test's dips, and round 4e
    (4,608 at 3e-6) lost every real-speech test set to round 4 (8,192), its data mix changed too
    ([plan](../vi_en_parity_plan.md#order), items 4 and 6). The square-root rule for Adam
    (Malladi et al., 2022) keeps the drift and noise per sample of data where gradient noise
    dominates; at 2.25e-6 the movement per hour of data is 1.33 times round 4's (weight decay's
    too). It is applied in part (the full rule also scales 1 − β1, 1 − β2 and ε; the betas stay
    (0.9, 0.95)) and is untested on this model.
  - Linear warmup over 5% of the steps (`--warmup-fraction`), cosine decay to 0 (`--schedule`
    cosine; the others are in [Schedules and averaging](#schedules-and-averaging)).
    The next run ends sooner, since the two-way run's last 12% of updates, the rate under 4% of
    its peak, gained nothing (user, 2026-11-13; [two-way](two-way.md#two-way-run)); `--max-steps`
    shortens the whole cosine, not only its tail. A floor held at the end instead would keep the
    model moving, as the constant 1e-6 before the cosine did
    ([plan](../vi_en_parity_plan.md#decisions-2026-11-01-later-dates-marked)). Fused AdamW with betas (0.9, 0.95)
    and weight decay 0.1; grad clip 1.0. Each train-log line holds its steps' mean and max gradient norm before clipping (`grad_norm`, `max_grad_norm`),
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

## Target timing

- A row's target starts at its cached delay or at `streams.start_floor`, whichever is later: no
  target text token starts before 0.32 s (`START_MARGIN`, 4 frames) after the source token a
  proportional alignment gives it has been heard (target token i of N waits for source token
  ⌈(i + 1)M / N⌉ of M; a source token is heard by the next one's start, the last 0.4 s after its
  own). A row's source token times are its pair's text track: the row of the same id in the other
  direction's cache (user, 2026-11-22: fix English to Vietnamese starting ahead of its English).
- Why: the cached delay counts from the source clip's start (U[0, 0.5 × its duration],
  `cache_rows.target_delay_s`) and ignores where its words are. VI→EN runs behind anyway (its
  English is longer and English clips open on silence), but EN→VI put a Vietnamese word over 1 s
  ahead of its English in 20–30% of rows, and began before the English speaker in 4–23%, on one
  shard of each source (2026-11-22). A model trained on that has to guess words not yet heard.
- What it moves, on one local shard of each source (58,482 rows of both directions, 2026-11-22):

  | Rows | EN→VI later | by (median, p90) | median start | VI→EN later | by | median start |
  |---|---:|---:|---|---:|---:|---|
  | Common Voice | 70% | 1.20, 2.40 s | 1.60 → 2.48 s | 17% | 0.40, 1.22 s | 1.12 → 1.20 s |
  | Libriheavy | 47% | 1.20, 3.12 s | 2.40 → 3.04 s | 24% | 0.56, 1.60 s | 2.00 → 2.16 s |
  | GigaSpeech 2 | 73% | 0.88, 2.08 s | 1.04 → 1.92 s | 42% | 0.40, 1.04 s | 0.88 → 1.04 s |
  | PhoAudiobook | 46% | 1.60, 3.84 s | 3.12 → 4.00 s | 16% | 0.72, 2.00 s | 2.64 → 2.80 s |
  | viVoice | 60% | 1.36, 2.88 s | 2.00 → 2.96 s | 17% | 0.48, 1.36 s | 1.76 → 1.76 s |

  No more than 0.04% of targets start after their source ends, as before.
- `load_rows` leaves out a row whose pair is missing: the other direction's build rejected it at
  alignment, so its speech and text disagree there (2.2% of those rows, most of them GigaSpeech 2's
  `vi-en` rejects). It also leaves out one whose Vietnamese, its own or its pair's, runs on more
  than 1.5 s past its last text token (`Aligner.max_tail_s`, 0.7%), the gate the build now applies
  ([cache-build](cache-build.md#rows)), so caches built before it lose those rows too. Its log line
  counts each reason and how many rows start later.
- The Vietnamese tail gate: VieNeu sometimes keeps speaking past its text, which the alignment
  score does not see. On Common Voice shard 22's `en-vi` rows (8,019), 109 (1.4%) are loud over
  0.6 s from 1 s past the last token; a tail over 1.5 s flags 2.4% of rows and catches all 109,
  and the greedy-CTC transcripts of 600 rows show babble where the tail is 5–11 s. Real Vietnamese
  recordings run under 0.72 s at p99 (>1.5 s ≤0.18%). A gate on the longest gap between tokens
  adds no babble row and hits real pauses (2.4% of PhoAudiobook rows have one over 2 s): not used.
  English is not gated: Common Voice clips end on recorded silence. Repeated words and names read
  letter by letter stay (CER over 0.25 on 12% of the 600; the alignment score tracks CER at
  r = −0.67).
- Every run reads both directions: `--cache-dir` for what it trains, `--pair-cache-dir` for the
  other direction's text alone (GRPO's `en_speech` `en-vi` rows pair with the two-way run's noisy
  `vi-en` copy, `grpo_launch.sh`); a shard none of whose rows has a pair raises. `validate.py`
  requires `--pair-cache-dir`. The floor and the left-out rows change every plan, the validation
  streams' too, so validation losses do not compare across the change, and `pair_cache_dir` is a
  resume key, so no run from before it resumes on it.

## Schedules and averaging

Runs keep stopping before their cosine ends (`student_dep10` at ~100,000 of 190,126 updates; its
first distillation at 32,000 with the rate still high, which then needed a separate anneal at
1e-5 to lower its KL, [phone-student](phone-student.md#distillation)). Warmup-stable-decay (WSD)
decides where to decay when the run is judged, not when it starts (2026-11-21; it matches cosine
at the same compute, Hägele et al. 2024, arXiv 2405.18392):

- `--schedule constant`: the warmup, then the peak until the run stops (by `--epochs` or
  `--max-steps`, or stopped by hand when validation stops improving).
- `--schedule decay`: a new run from one of the constant run's pairs, its model as
  `--model-weight` and its `trainer_stepNNNNNN.pt` as `--optimizer-state`
  (`checkpoints.load_optimizer_state`: the trainer file names its model, which must sit beside it
  under that name and be `--model-weight`, a hard link counting; AdamW's moments continue, the
  run's own betas and weight decay apply). It requires `--optimizer-state` and
  `--warmup-fraction 0`, since the moments need no warmup. The rate falls from the peak to 0 as
  1 − sqrt of the progress over the run's `--max-steps`. The pair is read only at the start, but a
  running constant run prunes it (`--keep-checkpoints`): hard-link both files, under their own
  names, into a directory of their own (`weights/<run>_stepNNNNNN/`) and branch from there. Give the
  decay a new `--seed` for fresh streams, as `student_dep10_anneal` had. Several decays can branch
  from different pairs of one constant run.
- **Averaging** (`python -m training.finetune.average --out OUT WEIGHT...`): the mean of a run's
  checkpoints (the same tensors and tokenizer; floats averaged in fp32), with the first one's
  metadata and `averaged_from`. Score it as any checkpoint: `eval.py` with the run's
  `--eval-pairs` and `--seed` on the box, or `gate.py` after `convert_mlx.py` on the Mac.
  Free-running chrF moves ~±0.7 between evaluations, so compare on the same rows and seed.
- **Validated on the box** (2026-11-21, the heads student's best at 176,000, distilled from GRPO
  update 1,400, 12,288-frame batches): a constant run at 1e-5 held the rate for 200 steps (FLEURS
  dev 38.03 / 32.55 at 200, from 38.34 / 35.64: ten times the rate its run ended on); a decay
  branch from its 200 pair (hard-linked) fell as 1 − sqrt (7.0e-6 at step 10, 3.0e-6 at 50,
  5e-8 at 100) to 38.40 / 34.35; the average of its 100 and 200 models is their exact mean on all
  135 tensors, keeps the metadata with `averaged_from`, and scored 38.59 / 34.43 (`eval.py`,
  seed 47).

## Speed

The fast path (`lm_training`), its measurements and the speed by GPU are in
[sft-speed](sft-speed.md).

## Objective

`common.compute_batch_losses`, Kyutai's `moshi-finetune` weighting (decided 2026-11-01,
[plan](../vi_en_parity_plan.md#decisions-2026-11-01-later-dates-marked)), as weighted sums of one kernel,
`common.token_ce`: each target token's cross-entropy over the full logits, 0 where the forward's
mask is off. Training, teacher-forced validation and GRPO's token logprobs (its negative at the
sampling temperature) all run it, so a change to it needs all three checked
([sft-speed](sft-speed.md#by-gpu) has why it is dense):

- Audio: the weighted mean CE over the 16 English target codebooks, codebook 0 (semantic)
  weighted `--first-codebook-weight` 100 against 1 for each acoustic codebook.
- Text: the weighted mean CE over every text position, a PAD weighted `--text-pad-weight` 0.5
  against 1 for a word token or EOS.
- Loss = 1.0 × audio + 1.0 × text. Logs and validation also report the semantic and acoustic
  CE and the word and PAD CE and accuracy unweighted.
- The upstream model on two 96-frame streams: loss 8.61, word CE 13.3, PAD accuracy 0.64. The
  previous objective is in [docs/analysis/loss_function.md](../analysis/loss_function.md).
- **Distillation** (`--teacher-weight`, [phone-student](phone-student.md#distillation)): each
  training token's CE becomes `common.token_kl`, its KL from a frozen bf16 teacher's
  distribution on the same stream, with the same weights, where both models' masks hold (a
  heads student's delays differ from its teacher's); the teacher reads the whole stream and the
  model its cut. Validation still reports CE against the targets.
- **Palettization** (`--palettize-bits`, a heads student): the step runs on the weights snapped
  to the Core ML export's k-means tables, the straight-through estimator
  ([student-heads](student-heads.md#palettization)).
- **Scheduled sampling** (`--sampled-text P`, not with a teacher): the main transformer samples
  every scored text token without gradients, each word piece it reads becomes the sampled word
  piece there with probability P, and the loss scores the dataset's tokens; the PADs (timing),
  each step's own inputs to the depformer and the audio stay the dataset's
  (`compute_batch_losses(..., history=)`, `lm_training.forward_history`); the train log adds
  `text_changed` ([exposure-bias](exposure-bias.md#scheduled-sampling)).

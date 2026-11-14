# Two-way speech: English to Vietnamese beside Vietnamese to English

Read when touching a direction (`--direction`, `cache_rows.Settings.direction`, a shard's or
pack's `direction`), the run's text tokenizer (`--tokenizer`, `common.run_tokenizer`,
`paths.swap_slots`, a pack's `tokenizer` and `text.bin`), the Vietnamese aligner
(`text_timing.ALIGNERS["vi"]`), an `en-vi` cache, the English-audio evaluation sets, the
direction condition (`training/finetune/direction.py`, `direction_conditions`), or the two-way run (`box/envi_en_speech.sh`, the `envi-cache` and `two-way` jobs; [plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Where a direction lives](#where-a-direction-lives) ·
[Tokenizer checks](#tokenizer-checks) · [Vietnamese aligner](#vietnamese-aligner) · [Checked](#checked) ·
[Two-way run](#two-way-run) · [Next](#next) · [Direction](#direction)

## Design

Decided 2026-11-10, for two-way training:

- A direction is `<source>-<target>`: `vi-en` (every cache built before 2026-11-10) or `en-vi`.
  The row layout holds in both: text, 16 target codebooks, 16 source codebooks
  ([sft-data-caches](sft-data-caches.md#row-layout)), so an `en-vi` row's target is Vietnamese
  speech and its text Vietnamese.
- An `en-vi` cache is built from audio through `cache_build`, never by swapping a `vi-en` row's
  tracks: the target text must be timed against the target audio
  ([cache-build](cache-build.md#rows)).
- `en-vi` text is tokenized with the Vietnamese swap (`tokenizer_spm_48k_multi6_2_vi6k.model`,
  [vi-tokenizer](vi-tokenizer.md)). It tokenizes `en_speech`'s English exactly as the base model
  does, so those `vi-en` caches stay valid under it and one run can mix both. Machine English
  is not covered by that check: a few rows of the tranche's `text_en` (GigaSpeech 2 3 of 19,967,
  viVoice 1 of 734, PhoAudiobook 0 of 1,299 in the first tranche) and of PhoMT (3 of 1,351 in
  shards 0–4) hold pieces whose slots now mean Vietnamese (`PRS` decodes as `Tùy`, `▁MCU` as
  `Khó`). A mixed run drops those rows ([below](#tokenizer-checks)).
- A stream holds one direction: `plan_streams` plans each direction apart (directions in name
  order, one RNG), so a mixed run's streams never switch language mid-stream. Switching within
  a stream comes later, if at all.
- Evaluation scores the text in both directions. Vietnamese ASR on the output audio is deferred.

## Where a direction lives

| Stage | How |
|---|---|
| cache rows | `Settings(direction=…)`: `.source`, `.target`, `.aligner` (the target language's); a builder's rows carry `text_en` and `text_vi` |
| shards | payload `direction` and `tokenizer` (its file name); samples keep `vi_frames` and `en_frames` (the target's counts its delay), both texts as the source wrote them, `text_timing` (the aligner's name); Vietnamese text is NFC'd before it is tokenized and timed |
| `cache_codes` | `--device cuda --direction en-vi` (MLX has no Vietnamese aligner); the tokenizer follows the direction (`paths.DIRECTION_TOKENIZERS`) |
| checkpoints | metadata `text_tokenizer`: the run's `--tokenizer` file name (none before 2026-11-10: the base) |
| trainers | `--tokenizer` defaults to the start weight's; each shard checked against it ([below](#tokenizer-checks)); `--val-cache-dir` takes each direction's cache, scored apart |
| tranche cache | `--direction en-vi`: `<root>/cache/en-vi/<source>/`, the cache repo's `en-vi/<source>/` (`cache_key`) |
| streams | `PACK_VERSION` 5 stores each pack's direction (a shard without one is `vi-en`) and tokenizer; `Row.direction`, `Piece.source_start`/`target_start` |
| model | the direction condition ([below](#direction)), from each stream's or evaluation row's direction |
| evaluation | a pairs file's audio column picks its direction (`common.EVAL_DIRECTIONS`: `vi_audio` → `text_en`, `en_audio` → `text_vi`); sets `fleurs-en`, `en-vi-val-en` ([sft-evaluation](sft-evaluation.md#evaluation-sets)) |

`mimi_cache_cuda` (the noisy re-encode) is `vi-en` only.

## Tokenizer checks

Every trainer and `eval.py`/`validate.py` check the run's `--tokenizer` against its inputs:

- A model weight (`common.run_tokenizer`, at startup in every entry point, before the caches
  load): its `text_tokenizer` metadata (`common.weight_tokenizer`; none is the base) must be the
  tokenizer's file name. `--tokenizer` defaults to that file under `weights/`, so a two-way
  checkpoint evaluates on the swap with no flag. A base weight reaches the swap only through
  `init_weights`. The swap's `.model` needs its `_init.json` beside it in `weights/`.
- A cache shard (`streams.load_rows`): its pack records the shard's `tokenizer` (none is the
  base). The same file loads whole. A shard on the base loads into a run on the swap
  without the rows whose text track uses one of the 6,000 slots the swap gave new pieces
  (`paths.swap_slots`, read from `<model>_init.json` beside the tokenizer): the rest decode the
  same on both. Anything else raises, so an `en-vi` shard never loads into a base run.
  Reassigned rows in round 4e's caches (2026-11-10, the box): 1,265 of 5,086,757 (0.025%):
  GigaSpeech 2 1,074 of 3,605,741, viVoice 167 of 260,139, PhoAudiobook 24 of 284,342, the 500
  noisy `en_speech` shards 0 of 936,535.
- A kept row decodes the same on the swap but keeps the base's split, not the swap's: in
  the same shards 1.6% of GigaSpeech 2's rows, 1.0% of PhoAudiobook's and viVoice's (Vietnamese
  names without diacritics, "Dak Lak", "Huynh Ngoc Tru"; 2–4 more pieces each), none of
  `en_speech`'s. VI→EN English then trains on the base's split of such a name and EN→VI on the
  swap's; both decode alike. Only a re-tokenized and re-timed `vi-en` cache would change it.
- Packs (`PACK_VERSION` 5) keep the rows' text tracks in `text.bin` for that check, so it
  reads ~3% of the codes. Old and new code repack each other's packs, so run one version on a
  cache folder at a time.
- Each `--val-cache-dir` gets its own stream plan and its own entry in `val_log.jsonl`'s
  `teacher_forced` (by its path), so a direction's losses stay apart and a `vi-en` cache's
  streams stay the same when an `en-vi` one joins (not across a planner change, such as late
  starts: [sft-training](sft-training.md)).

## Vietnamese aligner

`aadel4/omniASR-CTC-300M-v2` at `f2b3c39a…`: a stock-transformers conversion of Meta's
omniASR-CTC-300M-v2 (Apache-2.0), 325M parameters, a layer-norm feature extractor, a 10,288-letter
vocabulary of every script, blank `<s>`, words split by `" "`.

- Bake-off (2026-11-10, H200, 150 rows each of VieNeu TTS, PhoAudiobook and GigaSpeech 2):
  `nguyenvulebinh/wav2vec2-base-vietnamese-250h` (CC-BY-NC) and
  `MahmoudAshraf/mms-300m-1130-forced-aligner` (CC-BY-NC) against omniASR; every pair agreed on
  98.6–99.7% of word starts within 80 ms. omniASR is the one whose licence fits and the newest.
  Qwen3-ForcedAligner has no Vietnamese; `nguyenvulebinh/wav2vec2-base-vi-vlsp2020` needs a
  custom head.
- Its text (`_spoken_vietnamese`): NFC, lower case, numbers in Vietnamese words (`num2words`),
  every non-letter a word break. The alphabet is Vietnamese's 93 letters (12 vowels in 6 tones,
  the consonants, f j w z); a row with any other letter is rejected: 1 of 1,738 rows of a
  Libriheavy validation shard ("Teufelsbürst").
- Minimum score 0.5, calibrated on 400 rows a set (2026-11-10, H100): each clip aligned to its
  own text, to the text of the clip nearest in length, and to its own text with 30% of the words
  swapped for other rows' words. Share of each below 0.5:

  | Set | Own text | Another's text | 30% swapped |
  |---|---:|---:|---:|
  | VieNeu TTS (`en_speech` Libriheavy) | 0 | 98.2% | 0 |
  | PhoAudiobook | 0 | 98.8% | 0 |
  | viVoice | 0.3% | 99.2% | 0.7% |
  | GigaSpeech 2 | 0.5% | 99.2% | 2.5% |

  Own-text medians are 0.87–0.92; the wrong text's 95th percentile 0.44–0.47. English's
  wav2vec2-base-960h at the same 0.5 rejects 72.5% of another's text on Libriheavy (median 0.47
  against its own 0.96): the English gate is the looser one.
- Speed: the device hands the host only the Viterbi's columns (`vocab_columns`: blank,
  delimiter, alphabet). With all 10,288 a 13.5 h shard's log-probabilities came to ~100 GB copied
  and pickled to the workers: an `en-vi` Libriheavy validation shard built at 342× real time,
  1,638× with the columns (its `vi-en` build: 1,586–1,818×), every row equal.
- Batched against each clip alone (120 GigaSpeech 2 clips, TF32): the same word spans on all 118
  aligned rows, scores within 0.004.
- The Viterbi needs 2 × letters + 1 CTC frames (50 Hz). Vietnamese spends more of them than
  English (TTS validation rows: p99 0.86 of the frames, max 1.00, against English's 0.60–0.78),
  so fast real speech may lose rows as "CTC emission is too short": 3 of 400 GigaSpeech 2 rows
  failed alignment in the calibration. Read the rejections of the first `en-vi` tranche shards.
- For Vietnamese output, chrF and BLEU see tones; WER, exact match and the repetition gate fold
  text to ASCII (`common.ascii_text`), so they do not.

## Checked

On the H100 box, 2026-11-10, branch `twoway/prep`:

- A `vi-en` Libriheavy validation shard rebuilt by main and by the branch: the same 1,696 rows
  and rejections, every code and field equal; the payload adds `direction` and `tokenizer`.
- The validation cache's stream plan (9,313 rows, 1,568 streams) equals main's; with an `en-vi`
  shard added, 1,964 streams of one direction each, every row in one stream.
- `fleurs-en` and `en-vi-val-en` built in 54 s. `eval.py` on 40 rows each of `en-vi-val-en/cv27`
  and `fleurs-vi/dev` in shared batches, round 3's model moved onto the swap
  (`init_weights`): FLEURS chrF 44.5, healthy; English to Vietnamese chrF 10.1 (it answers in
  English: untrained).
- The tokenizer checks end to end (PR #100, the same box): a 30-step `train.py` on the swap with
  no `--tokenizer`, `vi-en` tranche shards and an `en-vi` shard, both directions' validation
  caches (dropped 7 reassigned rows; two `teacher_forced` entries; the checkpoint's metadata the
  swap; peak 74.9 GB); the refusals at startup (a weight on the other tokenizer, an `en-vi` shard
  in a base run, a repeated validation cache); a base run and `eval.py`, `validate.py` (its loss
  equal to the run's validation to the last digit), the packer and `grpo.py`/`grpo_profile.py`
  on the swap.

## Two-way run

The user's choices (2026-11-10): optimize the training first, use more data than the 400 h
pilot ("use more data for the two ways run"), update on round 4's 8,192 frames. So one run on
both directions of every row round 4e trained, and the pilot's test at its first validation.
Two box jobs ([tranche-box](tranche-box.md#jobs)):

- `restart.sh envi-cache` (~8 h): `fleurs-en` and `en-vi-val-en`; `box/envi_en_speech.sh`:
  the 3 `en_speech` validation shards and round 4e's 500 train shards (`box/en_speech_sub.txt`)
  by `cache_codes --direction en-vi` into `data/cache/en-vi/en-vi-grounded-v2/{train,validation}`,
  their sources fetched 25 at a time and deleted once built, then pushed to
  `en-vi/en-vi-grounded-v2/` of `huybik/hibiki-zero-vi-full-sft` (dataset; `upload_folder`, as
  huggingface-hub 2.4 has no `upload_large_folder`), failing unless the Hub lists every shard;
  then every tranche shard `en-vi` (`cache build --drop-inputs`, pushed to the cache repo's
  `en-vi/<source>/` as it goes). A rerun skips a shard built or packed.
- Then every cache keeps only its packs on the box: `python -m training.finetune.streams
  --drop-shards <dirs>`, the `vi-en` ones first (done 2026-11-10: 1,306 shards to version 5 packs,
  78.7 GB freed; the trainer then loaded 5,085,492 rows on the swap, the same 1,265 reassigned
  dropped), then the `en-vi` ones, each only once the Hub lists its shards (`--drop-shards` does
  not check, and `envi_en_speech.sh` counts a packed shard as built): both directions' shards and
  packs (~290 GB) would not fit the 350 GB disk beside a run's checkpoints. `two-way` refuses to
  start while a training cache holds shards.
- `publish_caches` names its Hub prefix after the out-dir's name, which for
  `data/cache/en-vi/en-vi-grounded-v2` is the `vi-en` cache's: never run it on that folder.
- The `en_speech` build restarts `cache_codes` every 25 shards (~35 s of model loading each,
  ~12 min over the 500); its single upload commit (~17 GB) has no retry, so a Hub error there stops
  the job before the tranche: rerun `restart.sh envi-cache`, which skips what is built.
- `init_weights` moves round 4's final model onto the swap
  (`weights/twoway_start_vi6k.safetensors`); `eval.py` scores it on the run's seven sets with
  `--seed 47`, batch 64: the baseline of the pilot's test.
- `twoway_sft` (`restart.sh two-way` then): both directions' caches (~8,500 h each), 8,192-frame
  batches computed in `--micro-frames` slices of 3,584, peak 3e-6, seed 47; validation every 12,000
  updates on both validation caches and `fleurs-vi/dev` and `fleurs-en/dev` (their mean selects),
  `multimed-vi/select`, `en-vi-val` and `en-vi-val-en`. The first validation is the pilot's test:
  EN→VI chrF up with healthy outputs, VI→EN FLEURS dev within 1 chrF of the start.
  `restart.sh two-way` now launches the SFT recipe instead (whole 4,608-frame batches at peak
  2.25e-6, validation every 20,000: [sft-training](sft-training.md#trainer)), so it no longer
  resumes `twoway_sft`: it starts `twoway_sft_4608`, at two epochs until its length is decided
  with the user (the next run ends sooner, [sft-training](sft-training.md#trainer)).
- The run (started 2026-11-11 04:32 UTC, 0.52 s an update, a validation ~7 min; on free-threaded
  Python from step 8,000, [environment](environment.md#python)): early on EN→VI rose fast while
  VI→EN fell (FLEURS dev 38.54 → 33.75 by 36,000) and climbed back later. The user extended it to
  two epochs (287,558 updates; sft-recovery.md: extending a run), selecting on the mean of both
  directions' FLEURS dev from step 32,000. Empty rows (no text token, no EOS, within the 20 s tail)
  are short clips (median ~7 s against 9-11 s); FLEURS-en's failed the health gate at times.
- The run ended 2026-11-13 02:39 UTC after 287,558 updates (every validation is in its
  `val_log.jsonl`). chrF, FLEURS-en's empty rows in brackets, against the start (`eval.py`,
  seed 47):

  | Step | Mean | FLEURS-vi | FLEURS-en | MultiMed | `en-vi-val` cv27 / libri | `en-vi-val-en` cv27 / libri |
  |---|---:|---:|---:|---:|---:|---:|
  | start | 22.46 | 38.54 | 6.37 (94) | 37.26 | 42.11 / 44.08 | 7.84 / 8.52 |
  | 96,000 | 27.15 | 33.69 | 20.61 (23) | 35.57 | 41.36 / 42.40 | 27.02 / 35.79 |
  | 120,000 | 29.29 | 36.42 | 22.15 (34) | 37.02 | 42.03 / 43.19 | 29.67 / 37.47 |
  | 156,000 | 30.04 | 37.78 | 22.29 (45) | 37.23 | 42.25 / 44.92 | 29.95 / 38.58 |
  | 168,000 | 29.64 | 36.46 | 22.82 (28) | 37.27 | 43.75 / 44.26 | 31.99 / 38.38 |
  | 192,000 | 27.98 | 34.34 | 21.62 (34) | 36.13 | 41.82 / 44.12 | 29.64 / 38.67 |
  | 228,000 | 29.60 | 35.88 | 23.32 (44) | 37.05 | 44.13 / 45.09 | 31.90 / 40.30 |
  | 252,000 | 30.51 | 37.09 | 23.93 (34) | 37.06 | 43.34 / 45.88 | 32.91 / 37.79 |
  | 287,558 | 30.13 | 37.22 | 23.03 (41) | 37.30 | 44.58 / 44.78 | 32.81 / 39.72 |

  - Promoted: 96,000, 120,000, 132,000, 144,000, 168,000, 240,000 and 252,000, the best. Every
    health failure was FLEURS-en's empty rows (41-66 against the 39 allowed, at 84,000, 108,000,
    156,000, 216,000, 228,000 and the final), at 156,000 and 216,000 with the highest mean yet.
  - The second epoch began at update 143,779. VI→EN dipped from 180,000 to 204,000 (the three
    Vietnamese-speech sets 1-2 chrF under 168,000) while both teacher-forced validation losses
    kept falling (VI→EN 2.677 → 2.671, EN→VI 2.575 → 2.560), and climbed back as the rate fell,
    as the learning-rate test's dips did ([plan](../vi_en_parity_plan.md#order), item 4).
  - The best came at 88% of the updates, the rate at 4% of its peak, and nothing after it scored
    higher (264,000 29.66, 276,000 30.27, the final 30.13): the last 12% gained nothing
    ([sft-training](sft-training.md#trainer)).
  - Against the start: EN→VI rose 6.37 → 23.93 on FLEURS-en and 7.8/8.5 → 32.9/37.8 on
    `en-vi-val-en`; VI→EN held on MultiMed (37.26 → 37.06) and rose on `en-vi-val` (+1.2/+1.8),
    but FLEURS dev ended 1.45 under the start, past the pilot's 1-chrF floor.
  - Rescored with the [input level stage](sft-evaluation.md#input-level) (2026-11-13, `eval.py`
    seed 47; seeds move a mean ~0.6), the checkpoints tie. Mean (FLEURS-en's empty rows):
    156,000 36.51 (5), 216,000 36.28 (5), 228,000 36.09 (3), 252,000 36.37 (4), the final 36.68
    (2; FLEURS-vi 39.63, FLEURS-en 33.73, MultiMed-ST 37.67, `en-vi-val` 43.58/45.43,
    `en-vi-val-en` 32.73/38.82). The final leads 252,000 by 0.31, inside the noise, so the
    shorter-run reading holds. The start scores 41.31 on FLEURS-vi with the stage (24.55 mean;
    FLEURS-en 7.79, 53 empty), so two-way costs VI→EN 1.7-2.0 on FLEURS dev, about twice the
    seed noise, while MultiMed-ST (37.33) and `en-vi-val` (43.56/43.58) hold or rise.
- Why a row stays silent: mostly a quiet source ([input level](sft-evaluation.md#input-level)).
  Half of FLEURS-en dev peaks under −40 dBFS: those 191 clips hold 31-61 of its empty rows at
  every validation from 36,000 to 72,000, the other 201 clips 4-9; FLEURS-vi's empty rows at
  60,000 and 72,000 all peak at −40 to −25 dBFS. Short clips go silent a little more often at any
  level (MultiMed-ST, 90% of it peaking above −12 dBFS: 0.6-1.4% of the clips under 8 s, 0-0.6%
  of longer ones), as the cache rarely starts a target after its source ends
  (`cache_rows.target_delay_s` draws the delay from U[0, 0.5 × the source's duration]). This
  run validates without the input level stage (its commit predates it), so its scores compare
  with each other, not with evaluations after 2026-11-12.

Build speed (2026-11-10, the H100 SXM, GigaSpeech 2 shards 0 and 1, both directions): `en-vi`
2,490× real time on the Mimi audio (omniASR 65 s on 42.8 h of Vietnamese, Mimi 72 s on 108 h),
`vi-en` 2,742× (wav2vec2-base 53 s, Mimi 71 s); the GPU thread is the bound. The Vietnamese
aligner keeps 15,843 of shard 0's 19,967 rows where the English one kept 16,574: ~730 more fail
alignment, most "CTC emission is too short" (fast speech).
`text_vi` is the corpus transcript: GigaSpeech 2's upper case lower-cased
([tranche-cache](tranche-cache.md)).

## Next

Two-way comes before GRPO (user, 2026-11-10, replacing that morning's "one way first"): round
4's final model is the VI→EN start (round 4e's test sets, [plan](../vi_en_parity_plan.md#order)),
then the [two-way run](#two-way-run) (done 2026-11-13), then one GRPO on both directions, from
the final model (`runs/twoway_sft/model_step287558.safetensors`; the promoted `best_step252000`
tied it under the input level stage). GRPO is done: its update 1,400 ships
([grpo](grpo.md#running)). Then the [direction](#direction) condition, from update 1,400.

## Direction

The shipping model (GRPO update 1,400) guesses the direction from the voice, and outside speech
fools it. On OneVoice's 100 EN→VI clips ([mlx-inference](mlx-inference.md#model-directories), 6
bits) 18 came out English, 12 of them on all of three more seeds; those 18 average −26 dBFS against
the rest's −21 (before the level stage) and are conversational or voice-assistant English ("no
that's all I needed help with, thank you" → "But Datta Niddedtowwess previously was well related to
Heave."); VI→EN wrote Vietnamese on 1 of 100. The user's own Vietnamese (15 s from the MacBook Pro
mic, file mode) came out Vietnamese on 5 of 5 seeds, heard right ("… bờ hồ … bình minh …"), while
OneVoice VI clips played through the laptop's speakers into the same mic came out English. Banning
non-ASCII text pieces gave English on 1 of 3 seeds, so the model's guess of the source wins over a
text mask. So the direction is a condition the model is trained on and the runtimes set (user,
2026-11-14, "go with A"):

- `training.finetune.direction` gives a checkpoint moshi's `lut` conditioner `direction` (entries
  `vi-en` and `en-vi`, 16 wide, projected to the model's 2,048 by a matrix that starts at zero),
  summed by the fuser into every frame's input, in its config.json, so every loader builds it. At
  zero the model's teacher-forced losses equal the parent's exactly (both validation caches, on the
  box); a projection drawn at N(0, 0.05) takes them from 2.7 to 13: the input is that sensitive.
- `hibiki_helpers.direction_conditions` hands it each row's direction, inside the model's graph:
  a stream's (`Stream.direction`, a batch's `directions`) in training, teacher-forced validation
  and GRPO's updates; an evaluation row's (`common.DIRECTION`, from its audio column) through
  `get_lmgen` in free-running evaluation and GRPO's rollouts. A model without conditioners (the
  upstream model, older checkpoints) gets none.
- `train.py --condition-lr` (1e-4) is the table's peak rate on the same warmup and cosine: at a
  fine-tune's rate it would barely move. GRPO keeps one group, so there the table moves at GRPO's
  rate, in effect frozen. `eval.py --direction` conditions every row on one direction (a model
  without the condition refuses it): the obedience test (flipped, the output language should
  follow it). `cut.py` refuses a config that would drop the parent's table.
- The run, `twoway_direction` (`restart.sh direction`, started 2026-11-14 04:59 UTC): update 1,400
  with the table, both directions' rows, 4,608-frame batches, the model at peak 1e-6 to keep GRPO's
  gains, 12,000 updates validated every 4,000.
- The before/after test, on the box: OneVoice's 200 clips (`data/eval/onevoice-{en,vi}/test.jsonl`)
  and `data/eval/user/test.jsonl` (`vi1`, `vi2`: the user's voice on the MacBook mic; `clean3`,
  `loop3`: OneVoice VI clips played through the laptop's speakers into it).

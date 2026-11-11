# Two-way speech: English to Vietnamese beside Vietnamese to English

Read when touching a direction (`--direction`, `cache_rows.Settings.direction`, a shard's or
pack's `direction`), the run's text tokenizer (`--tokenizer`, `common.run_tokenizer`,
`paths.swap_slots`, a pack's `tokenizer` and `text.bin`), the Vietnamese aligner
(`text_timing.ALIGNERS["vi"]`), an `en-vi` cache, the English-audio evaluation sets, or the
two-way run (`box/envi_en_speech.sh`, the `envi-cache` and `two-way` jobs; [plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Where a direction lives](#where-a-direction-lives) ·
[Tokenizer checks](#tokenizer-checks) · [Vietnamese aligner](#vietnamese-aligner) · [Checked](#checked) ·
[Two-way run](#two-way-run) · [Next](#next)

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
  streams stay round 4's when an `en-vi` one joins.

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
- `restart.sh two-way` (`twoway_sft`): both directions' caches (~8,500 h each), 8,192-frame
  batches computed in `--micro-frames` slices, peak 3e-6, seed 47; validation every 12,000
  updates on both validation caches and `fleurs-vi/dev` and `fleurs-en/dev` (their mean selects),
  `multimed-vi/select`, `en-vi-val` and `en-vi-val-en`. The first validation is the pilot's test:
  EN→VI chrF up with healthy outputs, VI→EN FLEURS dev within 1 chrF of the start.
- The run (started 2026-11-11 04:32 UTC, 0.52 s an update, a validation ~7 min; on free-threaded
  Python from step 8,000, [environment](environment.md#python)): at step 12,000 EN→VI rose
  (FLEURS-en 6.37 → 13.45, `en-vi-val-en` 7.8/8.5 → 14.5/20.8, Vietnamese but not yet faithful)
  while VI→EN FLEURS dev fell 38.54 → 36.73, 0.8 under the pilot's floor (MultiMed 37.26 → 36.09,
  `en-vi-val` flat). The user kept it running to recheck at 24,000, then extended it to two epochs
  (287,558 updates, ~41 h more; sft-recovery.md: extending a run). At 24,000 EN→VI kept rising
  (FLEURS-en 16.98, `en-vi-val-en` 20.4/28.0) and VI→EN kept falling (FLEURS dev 35.64, MultiMed
  35.67, `en-vi-val` cv27 40.49): the user chose two epochs selecting on both directions' FLEURS
  dev (the mean), applied from step 32,000: the restated `best.json` promotes step 24,000 (mean
  26.31 against 12,000's 25.09, both healthy).

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
then the [two-way run](#two-way-run), then one GRPO on both directions.

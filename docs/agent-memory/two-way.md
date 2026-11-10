# Two-way speech: English to Vietnamese beside Vietnamese to English

Read when touching a direction (`--direction`, `cache_rows.Settings.direction`, a shard's or
pack's `direction`), the run's text tokenizer (`--tokenizer`, `common.run_tokenizer`,
`paths.swap_slots`, a pack's `tokenizer` and `text.bin`), the Vietnamese aligner
(`text_timing.ALIGNERS["vi"]`), an `en-vi` cache, the English-audio evaluation sets, or the
two-way pilot ([plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Where a direction lives](#where-a-direction-lives) ·
[Tokenizer checks](#tokenizer-checks) · [Vietnamese aligner](#vietnamese-aligner) · [Checked](#checked) ·
[Next](#next)

## Design

Decided 2026-11-10, for the pilot:

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
  tokenizer's file name. `--tokenizer` defaults to that file under `weights/`, so a pilot
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

## Next

Two-way comes before GRPO (user, 2026-11-10, replacing that morning's "one way first"): after
round 4e's test sets pick the VI→EN start, the pilot, then the two-way SFT, then one GRPO on
both directions ([plan](../vi_en_parity_plan.md#order)). The pilot
([plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)): build an `en-vi` cache
subset of `en_speech` and the tranche (only one validation shard is built so far), start from a
model through `init_weights`, train ~10k mixed steps with both directions' validation caches
and pairs files. It passes if English-to-Vietnamese chrF rises with healthy outputs and FLEURS
dev stays within 1 chrF.

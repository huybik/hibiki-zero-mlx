# Two-way speech: English to Vietnamese beside Vietnamese to English

Read when touching a direction (`--direction`, `cache_rows.Settings.direction`, a shard's or
pack's `direction`), the Vietnamese aligner (`text_timing.ALIGNERS["vi"]`), an `en-vi` cache, the
English-audio evaluation sets, or the two-way pilot ([plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Where a direction lives](#where-a-direction-lives) ·
[Vietnamese aligner](#vietnamese-aligner) · [Checked](#checked) · [Next](#next)

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
  [vi-tokenizer](vi-tokenizer.md)). It tokenizes English exactly as the base model does, so the
  `vi-en` caches stay valid under it and one run can mix both.
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
| tranche cache | `--direction en-vi`: `<root>/cache/en-vi/<source>/`, the cache repo's `en-vi/<source>/` (`cache_key`) |
| streams | `PACK_VERSION` 4 stores each pack's direction (a shard without one is `vi-en`); `Row.direction`, `Piece.source_start`/`target_start` |
| evaluation | a pairs file's audio column picks its direction (`common.EVAL_DIRECTIONS`: `vi_audio` → `text_en`, `en_audio` → `text_vi`); sets `fleurs-en`, `en-vi-val-en` ([sft-evaluation](sft-evaluation.md#evaluation-sets)) |

`mimi_cache_cuda` (the noisy re-encode) is `vi-en` only.

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

## Next

The pilot ([plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)): an `en-vi` cache subset of
`en_speech` and the tranche, the start model through `init_weights`, ~10k mixed steps. It passes
if English-to-Vietnamese chrF rises with healthy outputs and FLEURS dev stays within 1 chrF.
Before it, the trainer should refuse a cache or model weight whose tokenizer is not `--tokenizer`
(neither is checked yet).

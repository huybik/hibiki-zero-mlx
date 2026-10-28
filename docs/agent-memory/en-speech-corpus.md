# English read-speech corpus

Read when touching the `en_speech` rows (`cv27-*`, `libriheavy-*` shards) of
`/Volumes/data/datasets/en_vi_speech`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Format](#format) · [Origin](#origin) · [Quality filters](#quality-filters) · [Rules](#rules)

## Purpose

This is the main training data source: real English speech with gold text, to be turned into
voice-matched VI→EN pair rows by the data phase of
[docs/vi_en_parity_plan.md](../vi_en_parity_plan.md). Its transcripts are all translated to
Vietnamese ([en-vi-translation](en-vi-translation.md)) and voiced in Vietnamese
([vi-tts](vi-tts.md)); no pair rows are built from them yet. Those rows would then go through
`cache_codes` and train together with the ~0.8k-hour PhoMT cache ([sft-data-caches](sft-data-caches.md#phomt-cache)).

## Contents

On 2026-10-23 the corpus moved into the EN-VI dataset
([en-vi-speech-dataset](en-vi-speech-dataset.md)), the one local copy: its shards keep their
names under `data/{train,validation}/`. The old `/Volumes/data/datasets/en_speech/` folder and
its 262 empty marker shards are gone.

| split | source | rows | hours | 5–15 s hours | speakers |
|---|---|---:|---:|---:|---:|
| train | Common Voice 27.0 EN (`cv27`) | 522,251 | 943.5 | 942.6 | 42,680 |
| train | Libriheavy small + medium + 1,688 of 2,955 large files | 2,958,303 | 9,542.2 | 7,217.6 | 4,459 |
| validation | Common Voice dev | 6,021 | 11.6 | 11.5 | 3,188 |
| validation | Libriheavy dev | 3,501 | 13.3 | 6.0 | 125 |

One Libriheavy train row, transcript `"."`, had nothing to translate and was dropped in the move.
Twelve more whose English clip is broken upstream are dropped, locally and on the Hub: 11 of
chapter 1 of *Elements of Conchology* (`libriheavy-large-00003-00075`) are 259-byte Ogg files with
headers and no audio, which stopped the TTS, and `mike_02_wodehouse_64kb_32`
(`libriheavy-medium-00016`) holds 5.7 s of its 21.2 s. A scan on 2026-10-25 of every Libriheavy
clip's Ogg length against `en_duration_s` found no others.

## Format

- Columns are the dataset's: `id`, `en_audio{bytes,path}`, `text_en` (cased, punctuated),
  `text_vi`, `speaker`, `en_duration_s`, then `vi_audio` and `vi_duration_s` (filled by the TTS
  step, [vi-tts](vi-tts.md)) and `cross_lingual_timbre_matched` (empty).
- Audio keeps the original codec bytes: 32 or 48 kHz mono MP3 for CV and 48 kHz mono Ogg Opus for
  Libriheavy. Resample to 24 kHz before Mimi.
- Audio bytes are uncompressed in the parquet and the other columns zstd. Row groups hold 500
  rows; CV shards hold up to 20,000 rows.
- Libriheavy ids (`libriheavy/<subset>/<speaker>/<book>/<recording>_<segment>`) and speakers
  carry a `libriheavy/` prefix; CV ids and speakers a `cv27/` prefix.

## Origin

`training/prep/build_en_speech.py` built the corpus and was retired once it moved into the
dataset; to rebuild, restore it from git (last at commit `7fde7f5`).

- Common Voice came from the Mozilla Data Collective API (dataset `cmu5jplf300nwmh07iqvk9leo`,
  `MDC_API_KEY`). Dev became validation; train is every other validated clip whose speaker and
  sentence are not in dev. The official CV test set is folded into train, so CV test scores are
  not held out.
- Libriheavy came from `mythicinfinity/libriheavy`: dev, medium and small, then large files in
  `random.Random(0)` order until pooled train held the target hours. About 10% of its rows
  are stereo and were dropped. A large file (~0.47 GB) added ~8.3 h after the filters.
- Every source keeps only clips of 5–22.4 s. A cache row holds each clip within the trainer's 280
  frames (22.32 s, one frame left for EOS), and the English delay is drawn within the room left
  ([sft-data-caches](sft-data-caches.md#flow)); a row is lost only when a clip itself is longer
  (454 English, 21.4k Vietnamese). Large files 195–1,688 in build order keep only 5–15 s clips (an
  earlier target).

## Quality filters

- Bandwidth uses HiFiTTS-2's estimate: the highest frequency whose Welch power is within 50 dB
  of the peak, 0 for silence. It is taken per recording: a Libriheavy chapter's first kept clips
  in one source file (clip id minus `_<segment>`) totalling ≥30 s, or a whole CV clip. Single
  clips of one chapter scatter ±1.5 kHz around the chapter estimate.
- Recordings under 10.5 kHz are dropped. Mimi codes up to 12 kHz. CV's 32 kHz MP3s cut off at
  ~11 kHz (a spike at 10.7–11.3 kHz) that an 11 kHz floor would split. Before the filter,
  42% of CV train hours and 17% of Libriheavy's were under 10.5 kHz.
- Train keeps a speaker's clips in build order (CV archive order, then Libriheavy medium, small
  and large) until the speaker holds 30 h; no CV speaker reaches it.
- The cap came from a simulation over the measured corpus and the footers of all 2,955 large
  files, counting 5–15 s hours as the target then did. Without it, 3,642 Libriheavy readers
  fill 9,200 h and one holds 261 h. 30 h left ~490 large files to spare; 20 h couldn't reach
  9,200 h.

## Rules

- The whole corpus is published with its translations in the EN-VI dataset
  ([en-vi-speech-dataset](en-vi-speech-dataset.md)): on 2026-10-23 the user chose to include
  Common Voice for research use, knowing the Mozilla Data Collective terms forbid re-hosting it.
  The dataset's `cv-anquach` rows are not corpus rows and miss these filters: Common Voice clips
  the corpus lacks, for a reason not recorded, re-encoded at 16 kHz
  ([en-vi-speech-dataset](en-vi-speech-dataset.md#cv-anquach-rows)).
- Training gets ~10k hours of 5–22.4 s English clips. The user first asked for 5–15 s clips; on
  2026-10-23 they chose 5–22.4 s and stopped the top-up at 10,486 train hours (8,160 of them
  5–15 s), knowing the trainer drops rows over 280 frames. On 2026-10-24 they dropped PhoMT
  from the EN-VI dataset and asked for 10k English hours; the corpus already held them, so they
  chose no top-up. Translation scope is in [en-vi-translation](en-vi-translation.md).
- Rows outside 5–22.4 s were deleted from the shards rather than rebuilt (user's choice), and
  large files keep only their clips. The quality filters were applied the same way on
  2026-10-23 (1.51M rows). Bringing deleted rows back means downloading them again.
- Train clip `cv27/common_voice_en_20040849` (9.4 s) was deleted on 2026-10-23, and the same
  sentence was removed from the translation selection. Sonnet returns an empty refusal for it,
  which stops `translate` ([en-vi-translation](en-vi-translation.md)). Rebuilding Common Voice
  from the archive would put the clip back.
- No other corpus is mixed in (user's choice, 2026-10-23, wanting many speakers, good
  transcripts and clean recordings). The English audio is a generated target, so 16 kHz
  corpora (LibriSpeech, GigaSpeech, People's Speech, VoxPopuli, Loquacious Set) lack the
  8–12 kHz band; upsampling adds none. Loquacious text has no case or punctuation, TED-LIUM 3
  is CC BY-NC-ND, GigaSpeech's HF terms are non-commercial, and HiFiTTS-2 is Libriheavy's own
  LibriVox audio. Emilia-YODAS English (CC BY, 24 kHz, spontaneous, unchecked Whisper-medium
  text) is the candidate for a spontaneous domain, with its own validation, once the first
  1k-hour tranche has trained.

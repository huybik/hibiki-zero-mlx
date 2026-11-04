# Vietnamese speech corpora

Read when choosing a Vietnamese speech corpus, checking its licence, or reading its files (layout,
labels, order, transcripts).
Back to [CONTEXT.md](../../CONTEXT.md) · [Corpora](#corpora) · [Corpus samples](#corpus-samples) ·
[YODAS layout and sample](#yodas-layout-and-sample). What the tranche takes from them, and how, is
in [real-vi-tranche](real-vi-tranche.md).

## Corpora

| Corpus | Hours | Content | Licence | Use |
|---|---:|---|---|---|
| YODAS2 vi, [`espnet/yodas2`](https://huggingface.co/datasets/espnet/yodas2)@`631fc44` (2.1 TB) | 14.7k (69k whole videos) | YouTube, mixed genres, 24 kHz, no speaker or dialect labels. Captions: ~0.78k h made by uploaders, ~9.2k h YouTube ASR; noisy, often unpunctuated, language labels unreliable | CC BY 3.0, per uploader | pool |
| Cleaned YODAS2, [`espnet/yodas_owsmv4`](https://huggingface.co/datasets/espnet/yodas_owsmv4)@`3bc62fd` | 4,644 vie | Segments of up to 30 s, re-aligned, wrong-language segments dropped; ids give the video and start/end in ms, so consecutive segments can be joined | CC BY 3.0 | first ~700 h |
| YODAS3 vi, [`espnet/yodas3`](https://huggingface.co/datasets/espnet/yodas3)@`81d802f` (670 GB) | 12.6k (44k videos, none in v1/v2) | ~95% YouTube ASR captions with word timestamps; whole videos | CC BY 3.0 | ~250 h, then the scale-up pool |
| Common Voice 27 vi (Mozilla Data Collective) | 23 (7.8 validated), 419 speakers | Read prompts | CC0 | speakers |
| FOSD, `doof-ferb/fpt_fosd`@`e6ff950` ([source](https://data.mendeley.com/datasets/k9sxg2twv4/4)) | 30 (the mirror's "100h" is wrong) | FPT 2018, sentences without punctuation | CC BY 4.0 | read speech |
| [PhoAudiobook](https://huggingface.co/datasets/thivux/phoaudiobook)@`f19f241` (VinAI, gated) | 941 (+554 augmented), 710 speakers | Audiobooks: book text, long continuous reading | Research or education only | first tranche |
| [viVoice](https://huggingface.co/datasets/capleaf/viVoice)@`693e988` (gated) | 1,017, 186 channels | Clean YouTube speech | CC BY-NC-SA 4.0 | first tranche, 5–22.4 s clips |
| [Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500)@`4b30571` (gated) | 500 | Mixed domains | CC BY-NC-SA 4.0 | not used ([samples](#corpus-samples)) |
| [GigaSpeech 2 vi](https://huggingface.co/datasets/speechcolab/gigaspeech2)@`8dc0d0e` (gated) | 6,039 refined | YouTube and podcasts, iteratively refined transcripts | Non-commercial research and education (access terms) | ~300 h first, then the scale-up pool |
| [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) | 15 | Read sentences | CC BY-NC-SA 4.0 | read speech |
| VLSP2020 100h, LSVSC 100h (dialect-tagged), VietSpeech 1,100h | 1.3k | Spontaneous, dialects, "social resources" | Only mirrors claim CC BY or Apache; primary terms or audio source unknown | after checking their terms allow research use |
| [ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) (102 h, 63 provinces); LEMAS vi | | | CC BY-NC-ND (no derivatives); LEMAS repackages GigaSpeech 2 and YODAS | not used |

- Emilia and Emilia-YODAS have no Vietnamese. Babel Vietnamese
  ([LDC2017S01](https://catalog.ldc.upenn.edu/LDC2017S01), 201 h of 8 kHz phone calls, four
  dialect regions) can be bought.
- Never train on the evaluation sets ([sft-evaluation](sft-evaluation.md#evaluation-sets)):
  - VietMed's labelled set is MultiMed-ST Vietnamese (the same 9,122 utterances and 3,437-row
    test set): `leduckhai/VietMed`, `doof-ferb/VietMed_labeled`, `leduckhai/MultiMed`, and the
    `vietmed_labeled` config of `nhatminh/vi-asr-relabel`. VietMed's unlabelled 2.2k h comes from
    the same TV channels.
  - FLEURS vi is inside `nvidia/Granary-v2` (its vi), the `fleurs_vi` config of
    `nhatminh/vi-asr-relabel` and `FluidInference/fleurs-full`.
  - YODAS may hold the same videos: drop every row whose transcript shares long n-grams with
    FLEURS vi or MultiMed-ST vi.

## Corpus samples

Measured 2026-11-04 on the box from one train shard each. The user accepted the four gated
corpora's terms for `huybik` that day.

- PhoAudiobook: parquet (`audio` as 16-bit PCM WAV at 16 kHz, though named float32, `text`,
  `speaker`), 803 train shards of 67–499 MB, 1,042,919 rows. From the names in every shard:
  - Shards 0–222 and the first 360 rows of 223 are the 290,037 original chunks
    (`chunk-<start>-<end>_trimmed_norm_float32.wav`, 947 h by their names, 10–20 s, p50 11.4 s).
    The rest, through shard 802, are the 752,882 "augmented short audio" clips (its card; 554 h,
    at most 10 s, cut inside the originals' spans), which ingest never downloads.
  - Each reader's chunks come as runs in rising time, one per reading, which carries no book id:
    17,030 runs of ~17 chunks, 213 of them crossing a shard boundary. The chunks sample a reading
    rather than cover it: the gap to the previous chunk is 63 s at p50 and 401 s at p90, and only
    15% of seams are within 1 s.
  - The names' times predate trimming: decoded audio differs from end − start by +0.09 s at p50,
    and 1,143 of 197,808 clips are over 1.5 s shorter (the worst 9.1 s; viVoice's differ by −0.38
    to +0.52 s), so seams come from alignment, not from the names.
  - Some readers are platforms (`Voiz`, `Fonos.vn`, `HTV`, `VOH`, `TV5050`) rather than one
    voice, and `Voiz` may be viVoice's `@VoizFM`: check for shared audio before using both.
  - Text is lower case with sentence punctuation, and chunks start and end mid-sentence.
- viVoice: parquet (`channel`, `text`, `audio` as 24 kHz WAV), 354 shards of ~480 MB.
  - Rows are shuffled across channels (169 in one shard) and carry no video id, so neighbours
    cannot be rejoined.
  - Clips are single sentences, cased and punctuated, mostly short (p5/p50/p95 1.0/3.3/9.8 s):
    55% of the hours, ~560 h of the corpus, are in 5–22.4 s clips.
- Bud500: 1.6–3.7 s fragments cut mid-phrase (1% of the hours in 5–22.4 s), lower case without
  punctuation, with no speaker or order. It cannot give rows of whole sentences, so it is dropped
  and GigaSpeech 2 takes its place.
- GigaSpeech 2 vi: `data/vi/train/N.tar.gz` (16 kHz WAV per segment, ~3.2 GB each; 828 GB for
  the three languages) and `train_refined.tsv` (5.0M segments, upper case, no punctuation).
  - Ids `<archive>-<recording>-<segment>` keep each recording's order, with gaps where refinement
    dropped segments: 120k recordings, with 15 segments at p50 and 117 at p90.
  - Its transcripts need the ASR's punctuation, and agreement between the two is a filter.
- FOSD: 48 kHz MP3, lower case without punctuation, p50 2.9 s (38% of the hours in 5–22.4 s).
  VIVOS: upper-case prompts, 46 train speakers, in one tar.

## YODAS layout and sample

Measured 2026-11-04 from the index files and one yodas2 shard (the sample is on the Mac under
`/Volumes/data/datasets/yodas_sample/`, with 20 cut clips in `cuts_owsmv4_on_yodas2/`):

- owsmv4 keeps Kaldi files for all 75 languages together under `dump/raw/yodas0.10/` (vie
  4,644 h, the cleaned subset) and `yodas0.00/` (vie 7,751 h, unfiltered): `text` (12.3 GB),
  `wav.scp`, `utt2spk`. Ids are `<YouTube id>_<start ms>_<end ms>_vie_asr`. Its audio is 16 kHz
  FLAC inside 500 `data_wav.ark` files of 36.8 GB, readable by HTTP range per segment.
- yodas2 vi is `data/{vi000,vi100,vi101}/audio/NNNNNNNN.tar.gz`: 1,437 shards of ~1.48 GB, each
  ~10 h of whole-video 24 kHz WAV, readable only whole. All 38,596 owsmv4 vie videos are yodas2
  videos, and owsmv4's segment times match the yodas2 audio within 0.4 ms. 4,324 of owsmv4's
  hours come from YouTube ASR captions (vi100/101), 320 from uploaders' (vi000).
- yodas3 vi: 80 WebDataset tars of ~8.6 GB (Opus `.webm`) and metadata parquets whose
  transcripts carry punctuation and, for auto captions, word times.

| owsmv4 vie | 0.10 | 0.00 |
|---|---:|---:|
| Segments / videos | 620,326 / 38,596 | 1,033,917 / 48,784 |
| Segment length p5 / p50 / p95 | 18.5 / 28.2 / 29.8 s | 18.0 / 28.3 / 29.9 s |
| Minutes per video p50 / p90 | 3.9 / 16.6 | 5.5 / 22.1 |
| Hours under a 20-minute cap per video | 3,825 | 5,914 |
| Gaps between a video's segments: 0 / ≤ 2 s / > 10 s | 49% / 65% / 29% | 69% / 88% / 5% |
| Contiguous hours in runs of 2–10 min / ≥ 10 min (gaps ≤ 2 s joined) | 1,684 / 227 | 3,305 / 2,947 |

- The segments are 25–30 s windows, not sentences: only 7.6% last 5–22.4 s, and their edges
  fall inside speech, often mid-phrase.
- The text is YouTube ASR: of 200 random segments 8 have any punctuation, casing is random,
  misrecognitions are common, and the median pace is 2.8 syllables/s (p5 1.2: some are not
  speech). Re-transcription is required.
- Of 20 clips, 5 clip at full scale and 6 have nothing above 8 kHz (upsampled sources).
- ~700 h costs ~185 GB of yodas2 shards (the best 125, 0.33 GB per kept hour in vi100/101,
  2.15 in vi000), streamed and cut on the box, or ~43 GB of 16 kHz FLAC by range reads from the
  owsmv4 arks.

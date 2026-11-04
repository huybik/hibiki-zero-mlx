# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its English TTS
or its licences.
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [Corpora](#corpora) ·
[YODAS sample](#yodas-layout-and-sample) ·
[English TTS](#english-tts) · [Pipeline](#pipeline)

Every Vietnamese source the model has trained on is VieNeu TTS. The channel (rooms, noise, phone
bands) costs ~5 of the 15–18 chrF between held-out TTS and FLEURS dev; the rest is real voices and
FLEURS's content, which only real Vietnamese speech can teach
([noisy-source](noisy-source.md#evaluation)). The tranche follows the paper's recipe for a new
language: real source speech, its transcript translated sentence by sentence, and the English
target synthesized in the speaker's voice (the paper's Italian: 850 h, base BLEU 14.3 → 31.4).
Researched 2026-11-04 from the corpora's and models' own cards; no code yet.

## Decision

- The first ~1,000 h come from YouTube video that uploaders marked CC BY (YODAS) plus Common Voice
  vi and FOSD, nothing non-commercial (user, 2026-11-04): the model ships in a commercial app.
- Before the model ships, a legal sign-off on:
  - YODAS's CC BY, which each uploader declared. Uploaders cannot be credited, because the
    dataset scrambles video ids, and some uploads are re-uploads of other people's content.
  - Cloning the voices of speakers who never consented. Several TTS cards forbid that. The
    existing data cloned only Common Voice (CC0) and LibriVox (public domain) readers.

## Corpora

| Corpus | Hours | Content | Licence | Use |
|---|---:|---|---|---|
| YODAS2 vi, [`espnet/yodas2`](https://huggingface.co/datasets/espnet/yodas2)@`631fc44` (2.1 TB) | 14.7k (69k whole videos) | YouTube, mixed genres, 24 kHz, no speaker or dialect labels. Captions: ~0.78k h made by uploaders, ~9.2k h YouTube ASR; noisy, often unpunctuated, language labels unreliable | CC BY 3.0, per uploader | pool |
| Cleaned YODAS2, [`espnet/yodas_owsmv4`](https://huggingface.co/datasets/espnet/yodas_owsmv4)@`3bc62fd` | 4,644 vie | Segments of up to 30 s, re-aligned, wrong-language segments dropped; ids give the video and start/end in ms, so consecutive segments can be joined | CC BY 3.0 | first ~700 h |
| YODAS3 vi, [`espnet/yodas3`](https://huggingface.co/datasets/espnet/yodas3)@`81d802f` (670 GB) | 12.6k (44k videos, none in v1/v2) | ~95% YouTube ASR captions with word timestamps; whole videos | CC BY 3.0 | ~250 h, then the scale-up pool |
| Common Voice 27 vi (Mozilla Data Collective) | 23 (7.8 validated), 419 speakers | Read prompts | CC0 | speakers |
| FOSD, `doof-ferb/fpt_fosd`@`e6ff950` ([source](https://data.mendeley.com/datasets/k9sxg2twv4/4)) | 30 (the mirror's "100h" is wrong) | FPT 2018, sentences without punctuation | CC BY 4.0 | read speech |
| VLSP2020 100h, LSVSC 100h, VietSpeech 1,100h | 1.3k | Spontaneous, dialect-tagged, "social resources" | Only mirrors claim CC BY or Apache; primary terms or audio source unknown | held until written terms |
| GigaSpeech 2 vi (6,039 h), Bud500 (500), viVoice (1,017), PhoAudiobook (941), VIVOS (15), ViMD (102), LEMAS vi (built from GigaSpeech 2) | ~8.6k | | Non-commercial | never in a shipped model |

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

### YODAS layout and sample

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

## English TTS

Kyutai's TTS 1.6B, the paper's, is out: its voice conditioning is closed and only precomputed
embeddings ship ([card](https://huggingface.co/kyutai/tts-1.6b-en_fr)). The finalists, with
Apache code and weights, at least 24 kHz, cloning from any reference clip:

| Model | Published English numbers | Notes |
|---|---|---|
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2B) | Seed test-en WER 1.84, similarity 75.3; 48 kHz | Vietnamese is one of its 30 languages, so a Vietnamese reference is in distribution; clones from the reference alone; batched serving through vLLM-Omni. Output varies between runs (its authors suggest 1–3 tries) |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | WER 1.24, similarity 0.775; 24 kHz | No Vietnamese; its best mode needs the reference's transcript |
| [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | WER 2.24, similarity 71.8; 24 kHz | A cross-lingual mode; the smallest and fastest |

- Ruled out:
  - non-commercial weights (F5-TTS, MaskGCT, Llasa, Spark-TTS, XTTS-v2, OpenAudio S1-mini, Fish S2);
  - licences that bar training other models on the output (IndexTTS-2 §3.4(c), Higgs Audio v2);
  - VibeVoice (research only, an audible disclaimer in every output);
  - MegaTTS 3 (its cloning encoder is unreleased);
  - OpenVoice v2 (22.05 kHz).
- Kyutai's `tts-0.75b-en-public` (CC BY 4.0, clones by an audio prefix, English only) is a fourth
  candidate if time allows: it is the closest to the paper's setup.
- None publishes Vietnamese→English speaker similarity, so a bake-off chooses, on the H100:
  1. 300 real Vietnamese clips from 150+ speakers (Common Voice and a YODAS sample), each with a
     5–10 s reference from another utterance of the same speaker and the clip's translation;
  2. each model's cloning modes at fixed seeds, every output round-tripped through Mimi (16
     codebooks), as training sees it;
  3. scores:
     - speaker similarity (WavLM-Large speaker verification) between the Vietnamese reference
       and the English output, anchored by same-speaker Vietnamese pairs (ceiling) and
       random-speaker pairs (floor);
     - Whisper-large-v3 WER on the English, and the share of broken clips: WER over 20%, or a
       duration per character far off;
     - throughput at batch 1, 16 and 64, giving H100-hours per 1,000 h;
  4. drop a model with more than 2% broken clips, take the highest similarity at WER ≤ 3% and
     at most ~50 H100-hours per 1,000 h, then listen blind to 20 pairs.

## Pipeline

Planned, and built for CUDA: every stage runs on the H100 box (user, 2026-11-04). The rows take
the EN-VI dataset's own parquet schema, and the cache keeps its format, so the trainer takes it
unchanged; the build needs a CUDA counterpart of the MLX `cache_codes` (Mimi in moshi's torch, as
`mimi_cache_cuda` encodes it, and the English CTC on CUDA)
([sft-data-caches](sft-data-caches.md#flow)):

```text
YODAS video audio (24 kHz)
  ─► Vietnamese ASR with punctuation and word times (the captions are ASR-grade)
  ─► sentences ─► rows of whole sentences, 5–22.4 s, kept in order per video
  ─► drop rows that overlap FLEURS vi or MultiMed-ST vi
  ─► LLM VI→EN per row, the video's previous rows as context (the existing translators)
  ─► English TTS in the row speaker's voice (reference: another stretch of the same speaker)
  ─► EN-VI parquet ─► CUDA cache build ─► trainer
```

- Rows come from re-transcribing contiguous runs, never from owsmv4's segments: choose videos
  whose 0.10 subset keeps ≥ 70% of the 0.00 speech (16,729 videos, 2,145 h under the cap), take
  their 0.00 runs (gaps ≤ 2 s joined), and spend each video's 20 minutes on one or two of its
  longest runs. Each row stores its `run_id` and `gap_before`, so the trainer joins rows only
  within a run.
- Filter rows on syllables per second, transcript confidence, language ID, clipping and
  bandwidth (flag sources at 16 kHz or below).
- YODAS has no speaker labels and a video may hold several speakers: cluster speaker embeddings
  per video to pick each row's reference and drop overlapped speech. The same embeddings and a
  dialect classifier check the North, Central and South balance.
- Rough cost from published speeds, to be measured: the TTS ~20–50 H100-hours per 1,000 h; the
  translation about the EN→VI runs' (~$39 per 1,000 h on Gemini,
  [en-vi-gemini](en-vi-gemini.md)).

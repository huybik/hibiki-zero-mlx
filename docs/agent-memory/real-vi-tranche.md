# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its English TTS
or its licences.
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [Corpora](#corpora) ·
[Corpus samples](#corpus-samples) · [YODAS sample](#yodas-layout-and-sample) ·
[English TTS](#english-tts) · [First tranche](#first-tranche) · [Models](#models) ·
[Pipeline](#pipeline)

Every Vietnamese source the model has trained on is VieNeu TTS. The channel (rooms, noise, phone
bands) costs ~5 of the 15–18 chrF between held-out TTS and FLEURS dev; the rest is real voices and
FLEURS's content, which only real Vietnamese speech can teach
([noisy-source](noisy-source.md#evaluation)). The tranche follows the paper's recipe for a new
language: real source speech, its transcript translated sentence by sentence, and the English
target synthesized in the speaker's voice (the paper's Italian: 850 h, base BLEU 14.3 → 31.4).
Researched 2026-11-04 from the corpora's and models' own cards; the code so far
(`training/tranche/`) is the English TTS bake-off, the ASR and translation checks, and ingest.

## Decision

- The tranche is research, so non-commercial corpora are in (user, 2026-11-04). Weights trained on
  them are research-only:
  - viVoice is CC BY-NC-SA and PhoAudiobook and GigaSpeech 2 research-only, and the recovery
    repo is public, so mark it and every checkpoint trained on them non-commercial (CC BY-NC-SA
    4.0) before the first such run.
  - Each source gets its own cache, so a commercial model is the same SFT without the
    non-commercial caches (~10 h of training); the generated data is kept either way.
- Storage (user, 2026-11-04: public, for research): every stage writes to the Hub shard by shard,
  `<source>/segments/`, `<source>/rows/` and `cache/<source>/`, so the box keeps only the shards
  in flight and the Mac can take a share.
  - [`huybik/hibiki-zero-vi-real`](https://huggingface.co/datasets/huybik/hibiki-zero-vi-real):
    public, gated with automatic approval on research-only, no-impersonation terms, CC BY-NC-SA
    4.0 overall; viVoice, YODAS, FOSD and VIVOS.
  - [`huybik/hibiki-zero-vi-real-restricted`](https://huggingface.co/datasets/huybik/hibiki-zero-vi-real-restricted):
    private; PhoAudiobook, whose user agreement forbids distributing it "in any original or
    modified form", and GigaSpeech 2, whose terms allow sharing only with people who accept
    them. Never make it public. Its first tranche (~1,100 h, ~37 GB) fits the free account's
    100 GB of private storage; a scale-up past that needs PRO (1 TB private).
- First tranche: ~2,000 h. The FLEURS and MultiMed lift per 1,000 h decides the scale-up, toward
  ~10k h with more GigaSpeech 2 and YODAS ([First tranche](#first-tranche)).
- A commercial build still needs a legal sign-off on:
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

### Corpus samples

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
| [dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar) (Apache-2.0; post-trained from dots.tts-base) | WER 1.30, similarity 77.1 | New; no serving stack |
| [LongCat-AudioDiT-3.5B](https://github.com/meituan-longcat/LongCat-AudioDiT) (MIT) | WER 1.50, similarity 78.6 | New; English and Chinese only |

- Ruled out:
  - non-commercial weights (F5-TTS, MaskGCT, Llasa, Spark-TTS, XTTS-v2, OpenAudio S1-mini, Fish S2);
  - licences that bar training other models on the output (IndexTTS-2 §3.4(c), Higgs Audio v2);
  - VibeVoice (research only, an audible disclaimer in every output);
  - MegaTTS 3 (its cloning encoder is unreleased);
  - OpenVoice v2 (22.05 kHz).
- Kyutai's `tts-0.75b-en-public` (CC BY 4.0, clones by an audio prefix, English only) is a fourth
  candidate if time allows: it is the closest to the paper's setup.
- From the cards and repos (2026-11-04): VoxCPM2 lists Vietnamese among its languages and
  dots.tts benchmarks it; Qwen3-TTS, CosyVoice3 and LongCat do not. LongCat is out: it needs the
  reference's transcript and was trained on Chinese and English only. No licence of the other
  four restricts training on their outputs. Only CosyVoice3 (Triton/TRT-LLM) and VoxCPM2 (Nano-vLLM
  or vLLM-Omni) document batched serving; their Python APIs, like dots.tts's, run one request at
  a time, and Qwen3-TTS takes lists.
- None publishes Vietnamese→English speaker similarity, so a bake-off chooses, on the H100
  (`training/tranche/bakeoff.py`, each model in its own uv env under `/workspace/envs`):
  1. real Vietnamese clips, one pair per speaker: PhoAudiobook readers, viVoice channels and
     GigaSpeech 2 recordings, each clip with another 5–16 s clip of the same speaker as its
     reference, and an `en-vi-val` English line to speak;
  2. each model's cloning modes at fixed seeds (VoxCPM2 reference-only and with the reference's
     transcript; dots.tts and Qwen3-TTS x-vector and with the transcript; CosyVoice3
     cross-lingual), every output round-tripped through Mimi (16 codebooks), as training sees it.
     Each model runs its best release: `dots.tts-soar` (post-trained; `-base` is its pretraining
     baseline) and CosyVoice3's RL-tuned LLM (`llm.rl.pt`, ranked above `llm.pt` on its card);
  3. scores:
     - speaker similarity (WeSpeaker ResNet293-LM: the seed-tts-eval WavLM-Large checkpoint is
       only on Google Drive) between the row's own Vietnamese clip and the English output,
       anchored by the clip against its reference (ceiling; rows under 0.4 are not one speaker)
       and against another speaker's clip (floor);
     - Whisper-large-v3 WER on the English, and the share of broken clips: WER over 20%, or a
       duration per character off the median by 2× either way;
     - the real-time factor at batch 1, one model at a time; batching only for the winner;
  4. drop a model with more than 2% broken clips, take the highest similarity at WER ≤ 3% and
     at most ~50 H100-hours per 1,000 h, then listen blind to 20 pairs.

## First tranche

~2,000 h, then measure:

| Source | Hours | Why |
|---|---:|---|
| PhoAudiobook, viVoice | ~1,300, capped per speaker or channel | The cleanest real speech with the most speakers; accurate transcripts where they come from text (audiobooks) |
| GigaSpeech 2 (recordings' segment runs) | ~300 | YouTube and podcasts, ordered within a recording |
| YODAS (owsmv4-selected videos) | ~400 | Spontaneous speech, many genres |
| Common Voice vi, FOSD, VIVOS | ~65 | More speakers, read sentences |

- Check each corpus's transcripts on a sample before trusting them; re-transcribe a corpus only
  where they are poor (always for YODAS).
- Keep a corpus's own order (an audiobook's consecutive sentences, a video's runs) so that the
  trainer can join real neighbours into streams.
- Rough box time per 1,000 h, from published speeds: the English TTS ~20–50 H100-hours, the
  ASR and the translation ~1 h each, alignment and diarization less: 1–3 days of the box,
  ~$35–100. Box time, not money, bounds the scale-up; past ~5k h a second GPU for generation
  pays.

## Models

The newest commercially usable model per step, from their cards (2026-11-04); every H100 speed
here is published or estimated, to be measured on the box first:

| Step | Pick (licence) | Runner-up | Notes |
|---|---|---|---|
| VAD | Silero VAD v6.2.3 (MIT) | FireRedVAD | |
| Diarization, overlap | `nvidia/Nemotron-3-Diarization` (OpenMDW-1.1) | pyannote community-1 (CC BY 4.0, gated) | DIHARD III DER 12.7 against 20.2; up to 8 speakers |
| Speaker embedding (voice reference) | WeSpeaker ResNet293-LM (CC BY 4.0) | TitaNet-Large | |
| Vietnamese ASR | `Qwen/Qwen3-ASR-1.7B` (Apache-2.0), vLLM | `nvidia/parakeet-ctc-0.6b-Vietnamese` (NVIDIA Open Model) | FLEURS-vi WER 5.55 against Whisper-v3's ~8.7; punctuated; ~1 h per 1,000 h at concurrency 128. Rows where the two ASRs disagree are dropped |
| Language ID | FireRedLID (Apache-2.0) | SpeechBrain VoxLingua107 ECAPA | Qwen3-ASR's own language tag cross-checks |
| Vietnamese word times (real and VieNeu audio) | NeMo Forced Aligner + Parakeet-vi | `omniASR_CTC_1B_v2` + `torchaudio.functional.forced_align` | Vietnamese accuracy unpublished: check on hand-timed rows |
| English word times | `Qwen/Qwen3-ForcedAligner-0.6B` (Apache-2.0) | the cache's `wav2vec2-base-960h` | Mean shift 37.5 ms against ~133; adopt only if its word starts agree with the existing caches' within a frame |
| VI→EN translation | `tencent/Hy-MT2-30B-A3B` (Apache-2.0 on its card since a licence change: pin the commit), vLLM on the box | `google/gemma-4-31b-it` (Apache-2.0) | FLORES XCOMET 93.85 against Gemini 3.1 Pro's 94.42; ~1 h per 10M words (estimate). Score ~200 of our sentences with xCOMET first |
| English TTS | VoxCPM2 | Qwen3-TTS 1.7B | Bake-off below |
| TTS bake-off scoring | WeSpeaker ResNet293-LM (scoring only) | WavLM-Large SV (its checkpoint is only on Google Drive) | |
| Dialect check | none usable exists | | A linear probe on WeSpeaker embeddings over ~300 hand-labelled speakers, plus the LLM tagging dialect words (mô/tê/răng; hông/nhen) |

- Ruled out, non-commercial or with licences that bar training on outputs: ChunkFormer-vie,
  hynt Zipformer, MMS-FA and MMS-LID, `wav2vec2-base-vi-vlsp2020`, Hunyuan-MT 1.x / HY-MT1.5,
  TranslateGemma.
- The Gemini, Claude and OpenAI terms forbid using outputs to build competing models, so open
  models on the box produce anything that becomes training data.
- For EN→VI, `pnnbao-ump/VieNeu-TTS-v3-Turbo` (Apache-2.0) is the newest open VieNeu.

## Pipeline

Planned, and built for CUDA: every stage runs on the H100 box (user, 2026-11-04). The rows take
the EN-VI dataset's own parquet schema, and the cache keeps its format, so the trainer takes it
unchanged; the build needs a CUDA counterpart of the MLX `cache_codes` (Mimi in moshi's torch, as
`mimi_cache_cuda` encodes it, and the English CTC on CUDA)
([sft-data-caches](sft-data-caches.md#flow)):

```text
YODAS runs (24 kHz) ─► Vietnamese ASR: punctuation, word times ─────────┐
other corpora ─► their transcripts, word times by forced alignment ─────┤ (ASR where they are poor)
                                                                        ▼
  rows of whole sentences, 5–22.4 s, in the source's order
  ─► drop rows that overlap FLEURS vi or MultiMed-ST vi
  ─► LLM VI→EN per row on the box, the previous rows as context
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
- Speaker labels cannot pick the TTS reference: YODAS has none, and GigaSpeech 2's recordings
  and some corpus "speakers" (platforms) hold several voices. On every source, cluster speaker
  embeddings per doc (per channel for viVoice, whose docs are single rows) to pick each row's
  reference and drop overlapped speech. The same embeddings and a dialect classifier check the
  North, Central and South balance.
- Ingest (`training/tranche/ingest.py`, CPU) writes each source shard's usable segments as one
  parquet sorted by `seq`, audio re-encoded to 32 kbps Ogg Opus (~16 MB per hour), under
  `/workspace/tranche/repo/<public|restricted>/<source>/segments/`, and `push` uploads that
  folder to its repo. `doc` is the recording as far as the corpus tells and never spans shards:
  a PhoAudiobook reader's run in rising time, a GigaSpeech 2 recording (its tars hold segments
  out of order, and with no times its neighbours are only consecutive segment numbers), and for
  viVoice the row itself. `speaker` is the corpus's label, none for GigaSpeech 2. A PhoAudiobook
  doc can join two readings when the second starts later than the first ended (190 gaps within
  a doc exceed 1 h), so translation takes context only across small gaps.
  One worker took 6 min for GigaSpeech 2's first archive (19,967 segments, 24.1 h);
  PhoAudiobook's first shard gave 1,299 clips (4.4 h), viVoice's 742 of its 2,508 (1.6 h; the
  rest outside 5–22.4 s).
- The box's 350 GB disk holds the noisy training cache (143 GB) and a run's checkpoint pair
  (37.5 GB, plus a second while it saves), so tranche data streams through it shard by shard.
  Build uv envs with `--no-cache`: the cache on `/workspace` grew ~35 GB in one env build and
  left 16 GB free, short of the LR test's next checkpoint save (2026-11-04).
- The box's huggingface_hub keeps file contents in a shared store, `$HF_HOME/hub/blobs/<xx>/<sha256>`,
  each with a `.refs` file naming the repo blobs that use it. Deleting a `models--…` or
  `datasets--…` folder frees nothing; delete a shared blob (with its `.lock` and `.refs`) once
  none of its refs exists (32 GB left behind by LongCat and the corpus samples, 2026-11-04).

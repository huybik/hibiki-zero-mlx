# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its English TTS
or its licences.
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [English TTS](#english-tts) ·
[First tranche](#first-tranche) · [Models](#models) · [Pipeline](#pipeline). The corpora, their
licences and measured layouts are in [vi-corpora](vi-corpora.md).

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
    them. Never make it public. Its segments take 21 GB for 1,277 h (~16 MB per hour), and
    the rows' English speech will about double that: the first tranche fits the free account's
    100 GB of private storage, and a scale-up past ~3,000 h needs PRO (1 TB private).
  - Each repo's card has a config per folder (`vivoice-segments`, `pho-segments`,
    `giga-segments`) declaring `audio` as Audio, so the viewer and `datasets` decode it.
- First tranche: ~2,000 h. The FLEURS and MultiMed lift per 1,000 h decides the scale-up, toward
  ~10k h with more GigaSpeech 2 and YODAS ([First tranche](#first-tranche)).
- A commercial build still needs a legal sign-off on:
  - YODAS's CC BY, which each uploader declared. Uploaders cannot be credited, because the
    dataset scrambles video ids, and some uploads are re-uploads of other people's content.
  - Cloning the voices of speakers who never consented. Several TTS cards forbid that. The
    existing data cloned only Common Voice (CC0) and LibriVox (public domain) readers.

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
  - Ingested and pushed on 2026-11-04 (`run_ingest.sh` on the box; each source's local copy
    deleted once pushed), and checked end to end against the sources:

    | Source | Rows | Hours | Labels | Docs | Hub | Time on the box |
    |---|---:|---:|---|---:|---|---|
    | PhoAudiobook (shards 0–223) | 290,037 | 939 | 710 readers | 17,243 | private, 15.7 GB | 23 min, 16 workers |
    | viVoice (5–22.4 s clips) | 269,350 | 571 | 186 channels | one per row | public, 9.7 GB | 30 min, 16 workers |
    | GigaSpeech 2 (archives 0–12) | 281,133 | 337 | none | per recording | private, 5.5 GB | 27 min, 4 workers |

  - The audio decodes to within 0.5 ms of `duration_s`. Clips under 1 s (some of GigaSpeech
    2's) decode 6.5 ms long, as ffmpeg cannot trim a one-page Ogg stream. Ogg serial numbers are
    random, so a re-ingested shard differs in bytes and uploads again.
- The box's 350 GB disk holds the noisy training cache (143 GB) and a run's checkpoint pair
  (37.5 GB, plus a second while it saves), so tranche data streams through it shard by shard.
  Build uv envs with `--no-cache`: the cache on `/workspace` grew ~35 GB in one env build and
  left 16 GB free, short of the LR test's next checkpoint save (2026-11-04).
- The box's huggingface_hub keeps file contents in a shared store, `$HF_HOME/hub/blobs/<xx>/<sha256>`,
  each with a `.refs` file naming the repo blobs that use it. Deleting a `models--…` or
  `datasets--…` folder frees nothing; delete a shared blob (with its `.lock` and `.refs`) once
  none of its refs exists (32 GB left behind by LongCat and the corpus samples, 2026-11-04).

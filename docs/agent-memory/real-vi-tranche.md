# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its models and
pipeline, or its licences. The corpora, their licences and measured layouts are in
[vi-corpora](vi-corpora.md); the English TTS is in [tranche-tts](tranche-tts.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [English TTS](#english-tts) ·
[First tranche](#first-tranche) · [Models](#models) · [Pipeline](#pipeline) · [Filters](#filters) ·
[Cache](#cache)

Every Vietnamese source the model has trained on is VieNeu TTS. The channel (rooms, noise, phone
bands) costs ~5 of the 15–18 chrF between held-out TTS and FLEURS dev; the rest is real voices and
FLEURS's content, which only real Vietnamese speech can teach
([noisy-source](noisy-source.md#evaluation)). The tranche follows the paper's recipe for a new
language: real source speech, its transcript translated sentence by sentence, and the English
target synthesized in the speaker's voice (the paper's Italian: 850 h, base BLEU 14.3 → 31.4).
Researched 2026-11-04 from the corpora's and models' own cards; the code so far
(`training/tranche/`) is the English TTS bake-off, the ASR check, ingest, translate, the
English TTS (`voice.py`), the row filters (`filters.py`) and the cache build (`cache.py`).

## Decision

- The tranche is research, so non-commercial corpora are in (user, 2026-11-04). Weights trained on
  them are research-only:
  - viVoice is CC BY-NC-SA and PhoAudiobook and GigaSpeech 2 research-only, and the recovery
    repo is public, so mark it and every checkpoint trained on them non-commercial (CC BY-NC-SA
    4.0) before the first such run.
  - Each source gets its own cache, so a commercial model is the same SFT without the
    non-commercial caches (~10 h of training); the generated data is kept either way.
- Storage (user, 2026-11-04: public, for research): every stage writes to the Hub shard by shard,
  `<source>/segments/`, `<source>/text_en/`, `<source>/en_audio/`, `<source>/asr_vi/` and `cache/<source>/`, so a rerun
  or the Mac picks up from the Hub. The box keeps what it wrote and downloaded (disk is ample).
  - [`huybik/hibiki-zero-vi-real`](https://huggingface.co/datasets/huybik/hibiki-zero-vi-real):
    public, gated with automatic approval on research-only, no-impersonation terms, CC BY-NC-SA
    4.0 overall; viVoice, YODAS, FOSD and VIVOS.
  - [`huybik/hibiki-zero-vi-real-restricted`](https://huggingface.co/datasets/huybik/hibiki-zero-vi-real-restricted):
    private; PhoAudiobook, whose user agreement forbids distributing it "in any original or
    modified form", and GigaSpeech 2, whose terms allow sharing only with people who accept
    them. Never make it public. Its segments take 21 GB for 1,277 h (~16 MB per hour), and
    the rows' English speech will about double that: the first tranche fits the free account's
    100 GB of private storage, and a scale-up past ~3,000 h needs PRO (1 TB private).
  - Each repo's card declares only the segments configs (`vivoice-segments`, `pho-segments`,
    `giga-segments`, `audio` as Audio), so the viewer and `datasets` decode those. `en_audio` has
    no config: its readers decode the Opus bytes themselves.
- First tranche: ~2,000 h. The FLEURS and MultiMed lift per 1,000 h decides the scale-up, toward
  ~10k h with more GigaSpeech 2 and YODAS ([First tranche](#first-tranche)).
- A commercial build still needs a legal sign-off on:
  - YODAS's CC BY, which each uploader declared. Uploaders cannot be credited, because the
    dataset scrambles video ids, and some uploads are re-uploads of other people's content.
  - Cloning the voices of speakers who never consented. Several TTS cards forbid that. The
    existing data cloned only Common Voice (CC0) and LibriVox (public domain) readers.

## English TTS

The English targets are voiced in each row's speaker's voice: the candidates, the bake-off that
chose VoxCPM2, its scores and running it are in [tranche-tts](tranche-tts.md).

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
  ASR ~1 h, the translation ~1 h if it runs on the box, alignment and diarization less: 1–3 days
  of the box, ~$35–100. Box time, not money, bounds the scale-up; past ~5k h a second GPU for generation
  pays.

## Models

The newest commercially usable model per step, from their cards (2026-11-04); every H100 speed
here is published or estimated, to be measured on the box first:

| Step | Pick (licence) | Runner-up | Notes |
|---|---|---|---|
| VAD | Silero VAD v6.2.3 (MIT) | FireRedVAD | |
| Diarization, overlap | `nvidia/Nemotron-3-Diarization` (OpenMDW-1.1) | pyannote community-1 (CC BY 4.0, gated) | DIHARD III DER 12.7 against 20.2; up to 8 speakers |
| Speaker embedding (voice reference) | WeSpeaker ResNet293-LM (CC BY 4.0) | TitaNet-Large | |
| Vietnamese ASR | `Qwen/Qwen3-ASR-1.7B` (Apache-2.0), vLLM | `nvidia/parakeet-ctc-0.6b-Vietnamese` (NVIDIA Open Model) | FLEURS-vi WER 5.55 against Whisper-v3's ~8.7; punctuated; ~1 h per 1,000 h at concurrency 128. `checks` (2026-11-05): WER 5.5% on FLEURS dev, 1.8% PhoAudiobook, 7.7% viVoice (3 of 300 clips detected as another language), 4.9% GigaSpeech 2; 941 clips in 11 s. Rows where the two ASRs disagree are dropped |
| Language ID | FireRedLID (Apache-2.0) | SpeechBrain VoxLingua107 ECAPA | Qwen3-ASR's own language tag cross-checks |
| Vietnamese word times (real and VieNeu audio) | NeMo Forced Aligner + Parakeet-vi | `omniASR_CTC_1B_v2` + `torchaudio.functional.forced_align` | Vietnamese accuracy unpublished: check on hand-timed rows |
| English word times | `Qwen/Qwen3-ForcedAligner-0.6B` (Apache-2.0) | the cache's `wav2vec2-base-960h` | Mean shift 37.5 ms against ~133; adopt only if its word starts agree with the existing caches' within a frame |
| VI→EN translation | Muse Spark 1.3 Contributor (`meta/muse-spark-1.3-contributor`) through OpenRouter, as the EN→VI text was ([en-vi-muse](en-vi-muse.md)) (user, 2026-11-05), for every source. Contributor mode lets Meta train on what it is sent, PhoAudiobook's and GigaSpeech 2's transcripts included; the user chose it over standard Muse Spark 1.3 (~17× the price, ~$170 for 2,000 h against ~$10) | `tencent/Hy-MT2-30B-A3B-FP8` on the box | A check on the box, since removed, measured Hy-MT2 (2026-11-05; vLLM 0.33 needs `flashinfer-jit-cache` beside it, or the FP8 MoE's first load compiles its kernels for about an hour): 21,000 rows (46.1 h) in 82–89 s, $0.013 per million characters against Muse's $0.084, but less faithful. FLEURS dev chrF 63.3 against Muse's 66.6. A blind Opus judge on 300 tranche rows gave it adequacy 3.90 against 4.44 (fluency 4.51 against 3.96), 26 rows at adequacy ≤ 2 against Muse's 8: it completes a segment's unfinished sentence and slips on pronouns, even when told to leave fragments unfinished (3.82 with the card's prompt). At ~$8 saved per 2,000 h, quality decides |
| English TTS | VoxCPM2, prompted by the row's own clip and transcript, checked by Qwen3-ASR and retried (`training/tranche/voice.py`) | dots.tts-soar | [bake-off and tuning](tranche-tts.md); chosen by blind listening |
| TTS bake-off scoring | WeSpeaker ResNet293-LM (scoring only) | WavLM-Large SV (its checkpoint is only on Google Drive) | |
| Dialect check | none usable exists | | A linear probe on WeSpeaker embeddings over ~300 hand-labelled speakers, plus the LLM tagging dialect words (mô/tê/răng; hông/nhen) |

- Ruled out, non-commercial or with licences that bar training on outputs: ChunkFormer-vie,
  hynt Zipformer, MMS-FA and MMS-LID, `wav2vec2-base-vi-vlsp2020`, Hunyuan-MT 1.x / HY-MT1.5,
  TranslateGemma.
- The Gemini, Claude and OpenAI terms forbid using outputs to build competing models, so open
  models on the box, or Muse, produce anything that becomes training data.
- For EN→VI, `pnnbao-ump/VieNeu-TTS-v3-Turbo` (Apache-2.0) is the newest open VieNeu.

## Pipeline

Planned, and built for CUDA: every stage runs on the H100 box (user, 2026-11-04), except the
translation, on Muse. The rows take
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
  ─► Muse VI→EN (OpenRouter) in batches of a doc's consecutive rows, which give each other context
  ─► English TTS in the row speaker's voice (prompt and reference: the row's own clip and transcript)
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
  embeddings per doc (per channel for viVoice, whose docs are single rows) to drop overlapped
  speech; the TTS needs no reference pick, as it prompts each row with its own clip. The same embeddings and a dialect classifier check the
  North, Central and South balance.
- Ingest (`training/tranche/ingest.py`, CPU) writes each source shard's usable segments as one
  parquet sorted by `seq`, audio re-encoded to 32 kbps Ogg Opus (~16 MB per hour), under
  `/workspace/tranche/repo/<public|restricted>/<source>/segments/`, and `push` uploads that
  folder to its repo. `doc` is the recording as far as the corpus tells and never spans shards:
  a PhoAudiobook reader's run in rising time, a GigaSpeech 2 recording (its tars hold segments
  out of order, and with no times its neighbours are only consecutive segment numbers), and for
  viVoice the row itself. `speaker` is the corpus's label, none for GigaSpeech 2. A PhoAudiobook
  doc can join two readings when the second starts later than the first ended (190 gaps within
  a doc exceed 1 h). Its rows are not contiguous speech: the corpus keeps parts of each reading,
  and consecutive rows of a doc start a median 40 s after the previous one ended (p90 4 min,
  three shards measured, 2026-11-05), so a stream joined from them has seams in time.
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
- Translate (`training/tranche/translate.py`, CPU, runs on the Mac) reads each segments shard's
  `id`, `doc` and `text` from the Hub, translates on Muse Spark 1.3 Contributor with the prompt
  the blind judge tested, in batches of up to 40 consecutive rows and 6,000 characters that end
  wherever a doc of more than one row starts or ends (viVoice's one-row docs share batches), and
  writes `<root>/<public|restricted>/<source>/text_en/<shard>.parquet` (`id`, `text_en`); `ingest
  push <source> --part text_en` uploads it beside the segments. A reply that does not parse, or
  that writes letters of a script its Vietnamese lacks (Muse wrote Chinese mid-sentence in 22 of
  the first 24,681 rows, "the keeper of the江湖 ledger"), splits its batch in halves down to one
  row; a row failing alone keeps `text_en` null, a rerun translates the null and broken rows
  again; a run stops at the first failed row that takes failures past 1% of the rows it has
  started (once past 5,000), and a lock keeps a second run off the same folder. A shard is
  written only once all its rows are done, so a crash loses the shards in flight (eight
  GigaSpeech 2 shards hold ~170k rows). Eight shards are in flight.
- Concurrency (2026-11-05): `--concurrency` defaults to 128. OpenRouter served 128 calls at once to
  Contributor with no 429s (two empty replies, retried); more is untested. Rows per call fall
  where docs are short (PhoAudiobook 11–25 a shard), and each call costs a few seconds beyond its
  rows, so short-doc sources need the concurrency, and a higher MAX_LINES would not help them.
  Measured on the full tranche:

  | Source | Rows | Concurrency | Calls | Time | Cost |
  |---|---:|---:|---:|---:|---:|
  | PhoAudiobook | 290,037 | 64 | 17,675 | 29 min | $3.86 |
  | GigaSpeech 2 | 281,133 | 128 | 11,400 | 8.8 min | $2.99 |
  | viVoice | 269,350 | 128 | 7,299 | 17 min | $2.90 |

  All 840,520 rows translated, none failed, $9.75; pushed and checked against the segments on
  the Hub (same shards and ids, no null or broken row) on 2026-11-05.
- The box's 350 GB disk holds the noisy training cache (143 GB) and a run's checkpoint pair
  (37.5 GB, plus a second while it saves), so tranche data streams through it shard by shard.
  Build uv envs with `--no-cache`: the cache on `/workspace` grew ~35 GB in one env build and
  left 16 GB free, short of the LR test's next checkpoint save (2026-11-04).
- The box's huggingface_hub keeps file contents in a shared store, `$HF_HOME/hub/blobs/<xx>/<sha256>`,
  each with a `.refs` file naming the repo blobs that use it. Deleting a `models--…` or
  `datasets--…` folder frees nothing; delete a shared blob (with its `.lock` and `.refs`) once
  none of its refs exists (32 GB left behind by LongCat and the corpus samples, 2026-11-04).

## Filters

`training/tranche/filters.py` measures what the cache build drops rows on; the build applies the
thresholds, so they can change without measuring again.

- Both filters compare text by `spoken`: lower-case NFC syllables without punctuation, digits
  written out by num2words' Vietnamese, and `ngàn`/`linh`/`nhăm` folded to `nghìn`/`lẻ`/`lăm`.
  PhoAudiobook and GigaSpeech 2 spell numbers; Qwen3-ASR and FLEURS write digits, and on the
  first PhoAudiobook shards rows with digits made 48% of those over 0.1 CER before (15 of 51
  digit rows over 0.1, 4 after). num2words reads a long digit run (a phone number) as one number.
- Evaluation overlap (`filters overlap`, Mac or box, text columns only): a row is dropped when its
  transcript holds a 13-syllable run of a FLEURS vi dev or test or MultiMed-ST vi select or final
  sentence, or a whole sentence of 9–12 syllables (62 of them) (`overlaps`, with the sets in
  `data/eval/`). On the first tranche no row does (2026-11-05). Runs of 10 syllables or fewer
  match stock broadcast phrases ("cảm ơn quý vị và các bạn đã quan tâm", 80 rows at 10, 257 at
  8), and the 18 sentences under 9 syllables ("được", "alo", "vùng nông thôn") everyday speech,
  so those are not matched.
- Vietnamese ASR (`filters asr`, box): every row heard by Qwen3-ASR-1.7B on the shared ASR
  server, language detected, not forced (unforced, it writes `language <Name><asr_text>…`; a reply
  without that header is all transcript with language `""`, as qwen_asr parses it), up to
  512 tokens, written as `<source>/asr_vi/<shard>` (`id`, `asr_vi`, `language`, `cer` against the
  corpus transcript over `spoken`) and pushed every 30 min and at each source's end. The
  rows where the two disagree, or the language is not Vietnamese, are the ones the build drops.
  - The client decodes each clip to 16 kHz WAV (4 processes): sent as Opus, the server's 8 API
    processes saturated the box's 27 cores at 21 rows/s and starved the TTS's main process.
  - Beside the English TTS on the same GPU it hears ~20 rows/s, bound by the engine (the TTS
    holds the GPU at 100%), and slows the TTS ~30% (a PhoAudiobook shard every ~2.0 min against
    ~1.55): ~12 h for 840,520 rows, about what running it after the TTS would take. Started
    2026-11-05 by `/workspace/restart_filters.sh` (log `/workspace/logs/filters-asr.log`).
  - First PhoAudiobook shards: CER median 0.000, p90 0.03–0.04, about one row a shard detected
    as another language.

## Cache

`training/tranche/cache.py build <sources>` (CUDA, the training `.venv`) turns each segments shard
whose `text_en`, `en_audio` and `asr_vi` are on the Hub into `cache/<source>/shard_<shard>.pt`
of its tranche repo, in `cache_codes`'s format ([sft-data-caches](sft-data-caches.md#row-layout)),
one cache per source (see [Decision](#decision)). A sample also holds `doc` and `seq`, so the
trainer joins a doc's rows in order ([sft-training](sft-training.md)); the delay draws use split
`train`.

- A row is dropped, with the first reason in the shard's `rejected`, when `en_broken`; when
  `en_asr` has Vietnamese letters `text_en` lacks; when its language is not Vietnamese; when its
  CER is over `--max-cer`; when the ASR hears more nghìn/triệu/tỷ than the transcript holds; when
  it overlaps an evaluation sentence; then on `cache_codes`'s own
  gates (decoded duration, frame cap, CTC alignment score 0.5, token timing).
- `--max-cer` 0.15 for every source (2026-11-06), from the CER of the first shards:

  | Source | Rows | p90 | p99 | Over 0.10 | Over 0.15 |
  |---|---:|---:|---:|---:|---:|
  | PhoAudiobook (16 shards) | 20,784 | 0.033 | 0.102 | 1.04% | 0.56% |
  | viVoice (30 shards) | 23,040 | 0.061 | 0.151 | 3.58% | 1.02% |
  | GigaSpeech 2 (2 shards) | 42,255 | 0.083 | 0.288 | 7.92% | 4.00% |

  Over 0.15 the transcript and the clip differ by phrases: PhoAudiobook transcripts that repeat
  the clip's last clause (0.17–0.34), punctuation spelled out ("chấm ba chấm"). The English
  translates the transcript, so such a row teaches the model to say what was not spoken. From
  0.10 to 0.15 the corpus is mostly right: on viVoice the ASR mangles English names and loanwords
  ("Jujutsu Kaisen" heard as "Shikai Sen", "Berserk" as "bơm sơn"), and percentages and year
  ranges are spelled otherwise ("30%" against "ba mươi phần trăm"); on PhoAudiobook a word at a
  cut edge or a number's spelling. Of 16 viVoice rows sampled there, one had a wrong fact ("dài
  200 tấn" where the speaker says "dài 282 mét"). 0.10 would drop 2.6% more of viVoice for that.
  From 0.05 to 0.10 the two differ by homophones (sinh/xin) and edge syllables. GigaSpeech 2's
  machine transcripts are the worst: over 0.15 its clips hold whole sentences more than their
  transcript (CER 1–1.9) or numbers lose their magnitude; from 0.10 to 0.15, of 20 rows sampled,
  5 write "một mươi" for "mười" (the same number), 5 leave out fillers ("cái"), and 5 have a wrong
  number.
- Magnitudes: GigaSpeech 2's transcripts drop nghìn, triệu and tỷ from numbers ("hơn một chuyến
  tàu" where the speaker says "hơn một nghìn chuyến", "ba ba trăm linh" for 3,300), at any CER. A
  row is dropped when the ASR (over `spoken`, so digits count) has more of them than the
  transcript: 2.06% of GigaSpeech 2's first 42,255 rows, 1.45% under the CER cut; 0.2% of
  PhoAudiobook's and viVoice's, mostly the ASR mishearing ("tết tỷ").
- Over the whole first tranche (`asr_vi` complete on 2026-11-06, 840,520 rows in 9.13 h), the
  gates that need no English drop, in their order:

  | Source | Rows | Language | CER over 0.15 | Magnitudes | Total |
  |---|---:|---:|---:|---:|---:|
  | PhoAudiobook | 290,037 | 0.01% | 0.48% | 0.17% | 0.66% |
  | viVoice | 269,350 | 0.03% | 0.99% | 0.17% | 1.20% |
  | GigaSpeech 2 | 281,133 | 0.03% | 3.91% | 1.39% | 5.34% |
- The Vietnamese-letter rule drops 118 of 10,392 rows on PhoAudiobook shards 0–7 (1.1%), 49 of
  them broken anyway. Of the rest, about half are lead-ins the TTS spoke in Vietnamese ("Không.
  Being rejected…", "Bước ngoặt khác, managing the farm") and half Vietnamese names the ASR spelled
  otherwise ("Tào Chí Viễn" for "Tào Trí Viễn"). Comparing words with their diacritics folded
  would keep 27 more (0.26%), not worth a second rule.
- CTC: transformers' wav2vec2-base-960h in float32 over padded batches, the first GroupNorm's
  statistics taken over each clip's own frames and the encoder given the attention mask, as
  `wav2vec2_mlx` masks. On CPU, 8 clips of a shard batched against each alone keep every argmax,
  their log-probabilities within 2.9e-3 (fp32 rounding: transformers' own fp32 and fp64 forwards
  differ by 3.1e-3); Mimi's batched codes equal the clip-alone ones on all 16 codebooks (moshi's
  Mimi is causal). Mimi as in `mimi_cache_cuda` (float32, TF32 off, 640 s batches). The Viterbi
  runs in the decode workers while Mimi encodes; the next shard is downloaded, read and gated in
  a thread, and its clips decode in the workers behind this shard's alignments.
- Needs `data/eval` (on the box `.venv/bin/python -m training.prep.eval_sets --sets fleurs
  multimed`, 41 s) and `weights/` ([docs/finetune.md](../finetune.md)). With the TTS and the ASR
  server on the GPU, ~1.3 GB of it is free, so the build runs after them.

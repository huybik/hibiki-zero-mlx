# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its models and
pipeline, or its licences. The corpora, their licences and measured layouts are in
[vi-corpora](vi-corpora.md); the English TTS is in [tranche-tts](tranche-tts.md), the cache build in
[tranche-cache](tranche-cache.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [English TTS](#english-tts) ·
[First tranche](#first-tranche) · [Models](#models) · [Pipeline](#pipeline) · [Filters](#filters)

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
    4.0) before the first such run. Marked 2026-11-06: the repo was CC BY-NC-SA 4.0 already, and
    its card now names the tranche runs (`tranche_epoch1/` on) research-only.
  - Each source gets its own cache, so a commercial model is the same SFT without the
    non-commercial caches (~10 h of training); the generated data is kept either way.
- Storage (user, 2026-11-06: every source public and gated, on the anquachdev account, the
  user's): every stage writes to the Hub shard by shard, `<source>/segments/`, `<source>/text_en/`,
  `<source>/en_audio/` and `<source>/asr_vi/` of `ingest.REPO`, and the caches to `<source>/` of
  `ingest.CACHE_REPO`, so a rerun or another machine picks up from the Hub. The box keeps what it
  wrote and downloaded under `/workspace/tranche/repo/<source>/<part>/` and `…/cache/<source>/`.
  - [`anquachdev/en-vi-speech-20k`](https://huggingface.co/datasets/anquachdev/en-vi-speech-20k)
    (`REPO`): PhoMT's synthetic pairs in `data/` (config `default`; the repo was
    `anquachdev/PhoMT-en-vi-speech` until 2026-11-06, and the old name redirects) and every part
    of the tranche but the caches. [`anquachdev/mimi-en-vi-speech-20k`](https://huggingface.co/datasets/anquachdev/mimi-en-vi-speech-20k)
    (`CACHE_REPO`): the caches, `<source>/shard_<shard>.pt`.
  - Both are gated with automatic approval: research-only, no-impersonation terms, each folder
    under its source's terms, and GigaSpeech 2's terms of access in full (their clause 4 lets a
    researcher give access to those who first agree to them; the user: our derived data complies).
    PhoAudiobook, whose user agreement forbids distributing it "in any original or modified form",
    is shared with its author's permission: the user asked the author and allowed it here
    (2026-11-06).
  - Pushes take `HF_TOKEN_AN`, a write token of anquachdev (`/workspace/.env` on the box; the jobs'
    scripts source it). Reads use huybik's own token, granted access through both gates. `push`
    refuses a repo that is not gated.
  - Why: huybik's free account has 100 GB of private storage, and the first tranche's private
    repo already used 56 GB (~44 MB per hour of GigaSpeech 2: segments 16, `en_audio` 20, cache 8).
  - The first tranche was in `huybik/hibiki-zero-vi-real` (public, gated: viVoice) and
    `huybik/hibiki-zero-vi-real-restricted` (private: PhoAudiobook, GigaSpeech 2 archives 0–12;
    never make it public). Both stay as they were; their files are copied to the new repos
    (2026-11-06: 143 MB/s over plain HTTP, 8 downloads at once with `HF_HUB_DISABLE_XET=1`; through
    Xet ~30 MB/s, and its threads hung after ~13 min). Re-uploading files the Hub already holds takes
    seconds: Xet deduplicates across repos (the first tranche's 591 caches in 20 s).
  - The card of `REPO` declares the segments configs (`pho-segments`, `vivoice-segments`,
    `giga-segments`, `audio` as Audio) beside PhoMT's `default`, so the viewer and `datasets`
    decode those. `en_audio` has no config: its readers decode the Opus bytes themselves.
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
| Vietnamese ASR | `Qwen/Qwen3-ASR-1.7B` (Apache-2.0), vLLM | `nvidia/parakeet-ctc-0.6b-Vietnamese` (NVIDIA Open Model) | FLEURS-vi WER 5.55 against Whisper-v3's ~8.7; punctuated; ~1 h per 1,000 h at concurrency 128. `checks` (2026-11-05): WER 5.5% on FLEURS dev, 1.8% PhoAudiobook, 7.7% viVoice (3 of 300 clips detected as another language), 4.9% GigaSpeech 2; 941 clips in 11 s. Parakeet was not run: the gate compares Qwen3-ASR with the corpus transcript ([tranche-cache](tranche-cache.md#gates)) |
| Language ID | FireRedLID (Apache-2.0) | SpeechBrain VoxLingua107 ECAPA | Not run: the gate uses Qwen3-ASR's own language tag ([tranche-cache](tranche-cache.md#gates)) |
| Vietnamese word times (real and VieNeu audio) | NeMo Forced Aligner + Parakeet-vi | `omniASR_CTC_1B_v2` + `torchaudio.functional.forced_align` | Vietnamese accuracy unpublished: check on hand-timed rows |
| English word times | `Qwen/Qwen3-ForcedAligner-0.6B` (Apache-2.0) | the cache's `wav2vec2-base-960h` | Mean shift 37.5 ms against ~133; adopt only if its word starts agree with the existing caches' within a frame. Not run: the tranche cache keeps `wav2vec2-base-960h`, as the existing caches do |
| VI→EN translation | `IndexTeam/Index-Translate-35B-A3B-preview` (Apache-2.0, Qwen3.5 MoE, FP8 as it loads) in vLLM 0.33 on the box, a row per prompt (`training/tranche/translate.py`), from GigaSpeech 2 archive 51 on (2026-11-06) | `IndexTeam/Index-Translate-9B-FP8` | [bake-off below](#vien-on-the-box). Before: Muse Spark 1.3 Contributor through OpenRouter (user, 2026-11-05) for the first tranche and GigaSpeech 2 archives 13–50, until Meta blocked the key on 2026-11-06 21:46 UTC+7 (HTTP 403 `user_blocked`, "repeated policy violations"); paid APIs (Gemini 3.1 Flash-Lite, ~$1.2 an archive on OpenRouter's batch price) were too dear for the user |
| English TTS | VoxCPM2, prompted by the row's own clip and transcript, checked by Qwen3-ASR and retried (`training/tranche/voice.py`) | dots.tts-soar | [bake-off and tuning](tranche-tts.md); chosen by blind listening |
| TTS bake-off scoring | WeSpeaker ResNet293-LM (scoring only) | WavLM-Large SV (its checkpoint is only on Google Drive) | |
| Dialect check | none usable exists | | A linear probe on WeSpeaker embeddings over ~300 hand-labelled speakers, plus the LLM tagging dialect words (mô/tê/răng; hông/nhen) |

### VI→EN on the box

Bake-off, 2026-11-06, vLLM 0.33 beside the ASR server, greedy, thinking off. FLEURS dev: its 149
sentences against their English. Archive 50: its first 2,000 rows, lowercased, against Muse's
English, both lowercased (Muse's misplaced lines count against every system). chrF/BLEU:

| Model, prompt | FLEURS dev | Archive 50 |
|---|---:|---:|
| Index-Translate-35B-A3B, a row per prompt, short | 65.1 / 37.7 | 70.5 / 42.6 |
| Index-Translate-9B, a row per prompt | 64.7 / 36.8 | 69.1 / 40.4 |
| Index-Translate-9B, a row per prompt, short | 64.8 / 36.7 | 68.7 / 40.1 |
| Index-Translate-9B, the card's prompt (Chinese, no context) | 63.6 / 35.4 | 64.4 / 34.4 |
| Index-Translate-9B, Muse's 40 numbered lines | 63.8 / 36.4 | 66.2 / 41.1 |
| Hy-MT2-30B-A3B, a row per prompt | 63.1 / 35.1 | |
| Hy-MT2-30B-A3B, Muse's 40 numbered lines (39 replies split) | 58.8 / 29.3 | 54.6 / 26.4 |

- "A row per prompt": the fragment instruction (translate exactly, leave an unfinished sentence
  unfinished), the doc's two earlier rows as context, names without diacritics, no brackets.
  Without the last two rules 8.3% of rows kept Vietnamese diacritics on names (Muse 0.05%), and
  a few added `[he]`. In 40-line batches the 9B moved words between neighbouring lines.
- On FLEURS, Gemini 2.5 Flash scored 66.8, Muse 66.6, Gemini 3.1 Flash-Lite 66.3, DeepSeek V3.2
  64.8 and Claude Haiku 4.5 63.6 (OpenRouter, Muse's 40-line prompt).
- Speed on archive 51 (19,707 rows), served over HTTP to a Python client: the 9B 70 rows/s
  (prefix caching on: each sequence's aligned state took ~1,000 tokens of cache and capped the
  batch near 345), 84 rows/s with it off; the 35B 110 rows/s. The GPU sat at 46% and ~300 W while
  the client's 512 threads held a core: in-process generation replaced the server.

- Ruled out, non-commercial or with licences that bar training on outputs: ChunkFormer-vie,
  hynt Zipformer, MMS-FA and MMS-LID, `wav2vec2-base-vi-vlsp2020`, Hunyuan-MT 1.x / HY-MT1.5,
  TranslateGemma.
- The Gemini, Claude and OpenAI terms forbid using outputs to build competing models, so open
  models on the box, or Muse, produce anything that becomes training data.
- For EN→VI, `pnnbao-ump/VieNeu-TTS-v3-Turbo` (Apache-2.0) is the newest open VieNeu.

## Pipeline

Built for CUDA: every stage runs on the H100 box (user, 2026-11-04; the translation too since
2026-11-06). The rows take
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
  ─► VI→EN on the box, a row per call with its doc's two earlier rows as context
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
  `/workspace/tranche/repo/<source>/segments/`, and `push` uploads that
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
- Each stage reads a shard's parts through `ingest.part_file`: the copy under `--root` where a
  stage on the box wrote one (all write whole, by a rename, and keep their files after pushing),
  else the Hub's. The box downloads through Xet at ~5 MB/s a stream (plain HTTP, `HF_HUB_DISABLE_XET=1`,
  reached 143 MB/s with 8 at once), and a second copy
  in the HF cache doubles a shard's disk. A part regenerated elsewhere and pushed loses to a stale
  local copy: delete the local one first (`text_en` refilled on another machine would leave voice
  stopping on "rows without text_en"). Translation still lists the Hub's segments and translates
  every shard without a local `text_en`: give it `--shards` past the shards translated elsewhere.
  Voice queues a shard only once its `text_en` is on the Hub or here.
- Translate (`training/tranche/translate.py`, in the env `/workspace/envs/mt` that `envs.sh`
  builds) runs Index-Translate-35B-A3B in vLLM in its own process. It reads each segments shard's
  `id`, `doc` and `text` from the Hub, translates each row as its own prompt, with up to two
  earlier rows of its doc as context, and writes `<root>/<source>/text_en/<shard>.parquet` (`id`,
  `text_en`); `ingest push <source> --part text_en` uploads it beside the segments.
  - It takes the GPU memory voice and the ASR server hold (0.9 of it), so translation and voice
    take turns: `/workspace/translate_burst.sh FIRST` (voice, filters and the ASR server stopped
    first) translates every pushed shard from FIRST on, pushes `text_en` and restarts them. Four
    shards go into each generate call, and the next four are read meanwhile.
  - Speed (2026-11-07, GigaSpeech 2, ~21,700 rows an archive): 204 rows/s, an archive in ~1.8 min,
    at 0.9 of the GPU, 512 sequences and 16k-token prefill steps; 125 rows/s at 0.75 beside the
    ASR server with 256 (PyTorch's allocator hitting out-of-memory). The GPU sits near 58% and
    430 W while vLLM's engine loop and the main process each hold a core: the engine is
    CPU-bound. Chat templates and tokenizing take 0.08 ms a row. py-spy cannot attach in this
    container (no ptrace).
  - GigaSpeech 2's transcripts are all capitals: a transcript all in capitals goes in lowercased,
    because kept, the model answers in capitals (Muse did on 5.7% of archive 50's rows), and such
    rows broke in the TTS 59.8% of the time against 21.3% (archives 13–24).
  - The first try is greedy with thinking off. A reply is tried again, sampled (temperature 0.7,
    top-p 0.8, top-k 20, a fresh seed each try: the engine's own is fixed), up to six tries in all
    (the later ones cover a few rows a group, in seconds), when it is cut at 512 tokens, empty, holds letters of a script its Vietnamese lacks, holds
    two of 28 frequent Vietnamese words, each with a diacritic (left in Vietnamese; names lose their
    diacritics, so "Nguyen Phu Trong" is right: ASCII words such as trong and khi stay out), or runs
    past 2.5× its Vietnamese's characters plus 20 (the model now and then translates the context
    rows too: 1 in 2,000 rows of the bake-off; English runs ~1.2× the Vietnamese).
  - A row failing every try keeps `text_en` null, is logged with its last reply, and holds its
    shard back from voice. A rerun tries only the null rows, sampled from the first try; a row
    already written is never translated again, as it may be voiced.
  - A run stops when more than half a group's rows fail a try, or once more than 1% of the rows it
    has tried fail every try (checked after each group, past 5,000); a lock keeps a second run off
    the same folder. Shards are written four at a time, once all their rows are done, so a crash
    loses the four in flight.
  - Muse Spark 1.3 Contributor through OpenRouter translated the first tranche and GigaSpeech 2
    archives 13–50 (2026-11-05/06), in batches of up to 40 consecutive rows: all 840,520 rows of
    the first tranche in ~55 min for $9.75. It wrote Chinese mid-sentence in 22 of the first
    24,681 rows ("the keeper of the江湖 ledger").
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
    ~1.55): ~12 h for 840,520 rows, about what running it after the TTS would take. It runs as
    `/workspace/restart.sh filters` (log `/workspace/logs/filters.log`).
  - First PhoAudiobook shards: CER median 0.000, p90 0.03–0.04, about one row a shard detected
    as another language.
  - Checked end to end on the box (2026-11-05), beside the running jobs: `asr pho --limit 1`
    into a scratch `--root`, its push stubbed, heard shard 0 in 79 s. The shard holds the
    segments' 1,299 ids in order, `id`/`asr_vi`/`language` as strings and `cer` as float32, no
    nulls; CER recomputed over `spoken` matches every row; a rerun skips it and pushes what is
    unpushed. Against the running job's own pass over the shard, 1,239 transcripts are
    identical and 60 differ (48 in punctuation or case only, the rest by at most 0.042 CER):
    the server's batching. The reply parser reads the five reply shapes (a language header,
    `language None`, an empty transcript, no header). The job's first push put 30 shards in the
    private repo, byte-identical to the local ones. `filters overlap` on the box finds no row
    of the 840,520; `overlaps` matches a 13-syllable run and a whole 9–12-syllable sentence, not
    a 12-syllable run or shorter sentences.
  - The highest CERs are faults of the corpus: PhoAudiobook transcripts that repeat the clip's
    last clause (0.24–0.34), numbers and punctuation spelled oddly ("chấm ba chấm", "triệu
    bốn nghìn"), abbreviations ("tp. hcm"); then words the ASR loses at a clip's cut edges.
    `spoken` leaves `%` and decimal commas unspoken (the ASR's "80%" against the corpus's "tám
    mươi phần trăm"), which adds at most ~0.04 CER to ~0.5% of rows, so it stays.

# Real-Vietnamese tranche

Read when choosing or building the real-Vietnamese source data (plan stage 1.3), its models and
pipeline, or its licences. The corpora, their licences and measured layouts are in
[vi-corpora](vi-corpora.md); the translation is in [tranche-translate](tranche-translate.md), the
English TTS in [tranche-tts](tranche-tts.md), the cache build in [tranche-cache](tranche-cache.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [English TTS](#english-tts) ·
[First tranche](#first-tranche) · [Models](#models) · [Pipeline](#pipeline) · [Filters](#filters)

Every Vietnamese source the model has trained on is VieNeu TTS. The channel (rooms, noise, phone
bands) costs ~5 of the 15–18 chrF between held-out TTS and FLEURS dev; the rest is real voices and
FLEURS's content, which only real Vietnamese speech can teach
([noisy-source](noisy-source.md#evaluation)). The tranche follows the paper's recipe for a new
language: real source speech, its transcript translated sentence by sentence, and the English
target synthesized in the speaker's voice (the paper's Italian: 850 h, base BLEU 14.3 → 31.4).
Researched 2026-11-04 from the corpora's and models' own cards. The code (`training/tranche/`) is
YODAS's ingest (`ingest.py`, `yodas.py`), translate, VieNeu's TTS both ways (`voice.py`), the row
filters (`filters.py`) and the cache build (`cache.py`). PhoAudiobook, viVoice and GigaSpeech 2 are
ingested and heard whole: their ingest and ASR pass, the bake-off and the ASR check are in git history.

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
  - Some plain-HTTP connections from the box crawl at ~50 KB/s and never time out (2 of 4 in a test,
    2026-11-06), and `hf_hub_download` waits on them: a copy sat 2 h on one file. Bulk copies
    download with curl, `--speed-limit 1000000 --speed-time 30 -C -` in a retry loop, which drops
    such a connection and resumes on a new one: `common-vi-en`'s 447 GB in 80 min, 8 files at once,
    sha256 checked before each commit. Uploading from the Mac's own copy instead saves no download:
    global dedup still leaves a few percent of each file to send (19 MB of a 500 MB shard), more
    than its uplink carried (18 Mbps, Xet uploads timing out, 2026-11-07).
  - The card of `REPO` declares the segments configs (`pho-segments`, `vivoice-segments`,
    `giga-segments`, `yodas-segments`, `audio` as Audio) beside PhoMT's `default`, so the viewer and `datasets`
    decode those. `en_audio` has no config: its readers decode the Opus bytes themselves.
- First tranche: ~2,000 h. The FLEURS and MultiMed lift per 1,000 h decides the scale-up, toward
  ~10k h with more GigaSpeech 2 and YODAS ([First tranche](#first-tranche)).
- A commercial build still needs a legal sign-off on:
  - YODAS's CC BY, which each uploader declared. Uploaders cannot be credited, because the
    dataset scrambles video ids, and some uploads are re-uploads of other people's content.
  - Cloning the voices of speakers who never consented. Several TTS cards forbid that. The
    existing data cloned only Common Voice (CC0) and LibriVox (public domain) readers.

## English TTS

The English targets are voiced in each row's speaker's voice, by VoxCPM2 for the first tranche and VieNeu-TTS v3
Turbo from YODAS's shard 0004 on, the voicer that now runs both ways: [tranche-tts](tranche-tts.md).

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
  ASR ~1 h, the translation ~1 h if it runs on the box, alignment less: 1–3 days
  of the box, ~$35–100. Box time, not money, bounds the scale-up; past ~5k h a second GPU for generation
  pays.

## Models

The models each step runs: the ASR, the translation and the TTS chosen from their cards (2026-11-04
on, the newest commercially usable) and measured on the box; the aligners are the cache build's.

| Step | Model (licence) | Notes |
|---|---|---|
| Vietnamese ASR | `Qwen/Qwen3-ASR-1.7B` (Apache-2.0), vLLM | FLEURS-vi WER 5.55 against Whisper-v3's ~8.7; punctuated. `checks.py` (2026-11-05, in git history): WER 5.5% on FLEURS dev, 1.8% PhoAudiobook, 7.7% viVoice (3 of 300 clips detected as another language), 4.9% GigaSpeech 2; 941 clips in 11 s. Its own language tag is the language ID ([tranche-cache](tranche-cache.md#gates)) |
| Vietnamese word times (real and VieNeu audio) | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` (CC BY-NC 4.0), `cache.timing.ALIGNERS["vi"]` | [two-way](two-way.md#vietnamese-aligner) |
| English word times | `facebook/wav2vec2-base-960h`, as the existing caches | `Qwen/Qwen3-ForcedAligner-0.6B` (mean shift 37.5 ms against ~133) not adopted |
| VI→EN translation | `IndexTeam/Index-Translate-35B-A3B-preview` (Apache-2.0, Qwen3.5 MoE, FP8 as it loads) in vLLM on the box, a row per prompt (`training/tranche/translate.py`), from GigaSpeech 2 archive 51 on (2026-11-06) | [bake-off](tranche-translate.md#bake-off). Before: Muse Spark 1.3 Contributor through OpenRouter (user, 2026-11-05) for the first tranche and GigaSpeech 2 archives 13–50, until Meta blocked the key on 2026-11-06 21:46 UTC+7 (HTTP 403 `user_blocked`, "repeated policy violations"); paid APIs (Gemini 3.1 Flash-Lite, ~$1.2 an archive on OpenRouter's batch price) were too dear for the user |
| English TTS | VieNeu-TTS v3 Turbo from the row's own clip, checked by Qwen3-ASR (`training/tranche/voice.py`); VoxCPM2 with the clip's transcript before | [choice and checks](tranche-tts.md): VoxCPM2 by blind listening, VieNeu for 3.1× its speed |
| TTS bake-off scoring | WeSpeaker ResNet293-LM (scoring only) | |

- Weighed and never built: a VAD (Silero VAD v6.2.3), diarization (`nvidia/Nemotron-3-Diarization`),
  speaker clustering per doc, a dialect check, and filters on pace, clipping or bandwidth. The rows'
  filters are the ASR gates and the evaluation overlap ([Filters](#filters)).
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
unchanged; the shared cache build runs it on CUDA, or on the Mac in MLX
([cache-build](cache-build.md)):

```text
YODAS3 caption runs (24 kHz) ─► Qwen3-ASR transcripts, gated on CER ───┐
other corpora ─► their transcripts, checked against Qwen3-ASR ──────────┤
                                                                        ▼
  rows of whole sentences, 5–22.4 s, in the source's order
  ─► drop rows that overlap FLEURS vi or MultiMed-ST vi
  ─► VI→EN on the box, a row per call with its doc's two earlier rows as context
  ─► English TTS in the row speaker's voice (reference: the row's own clip)
  ─► EN-VI parquet ─► CUDA cache build (word times by forced alignment) ─► trainer
```

- YODAS3's rows are cut from its videos' caption runs and transcribed by Qwen3-ASR during ingest
  ([tranche-yodas](tranche-yodas.md)), not taken from owsmv4's segments.
- Speaker labels cannot pick the TTS reference: YODAS has none, and GigaSpeech 2's recordings
  and some corpus "speakers" (platforms) hold several voices. The TTS needs no reference pick, as
  it prompts each row with its own clip.
- Ingest writes each source shard's usable segments as one parquet sorted by `seq`, audio
  re-encoded to 32 kbps Ogg Opus (~16 MB per hour), under
  `/workspace/tranche/repo/<source>/segments/`, and `ingest push` uploads that folder to its repo.
  `doc` is the recording as far as the corpus tells and never spans shards: a PhoAudiobook reader's run in rising time, a GigaSpeech 2 recording (its tars hold segments
  out of order, and with no times its neighbours are only consecutive segment numbers), and for
  viVoice the row itself. `speaker` is the corpus's label, none for GigaSpeech 2. A PhoAudiobook
  doc can join two readings when the second starts later than the first ended (190 gaps within
  a doc exceed 1 h). Its rows are not contiguous speech: the corpus keeps parts of each reading,
  and consecutive rows of a doc start a median 40 s after the previous one ended (p90 4 min,
  three shards measured, 2026-11-05), so a stream joined from them has seams in time.
  - The first tranche, ingested on 2026-11-04 and checked end to end against the sources (GigaSpeech
    2's other archives followed: all 240 are in the repo):

    | Source | Rows | Hours | Labels | Docs | Size | Time on the box |
    |---|---:|---:|---|---:|---|---|
    | PhoAudiobook (shards 0–223) | 290,037 | 939 | 710 readers | 17,243 | 15.7 GB | 23 min, 16 workers |
    | viVoice (5–22.4 s clips) | 269,350 | 571 | 186 channels | one per row | 9.7 GB | 30 min, 16 workers |
    | GigaSpeech 2 (archives 0–12) | 281,133 | 337 | none | per recording | 5.5 GB | 27 min, 4 workers |

    GigaSpeech 2 archives 86 and 98 (`ingest.SOURCES["giga"].heldout`) are held out of training
    as the tranche's own evaluation set ([giga-heldout](sft-evaluation.md#evaluation-sets)).

  - The audio decodes to within 0.5 ms of `duration_s`. Clips under 1 s (some of GigaSpeech
    2's) decode 6.5 ms long, as ffmpeg cannot trim a one-page Ogg stream. Ogg serial numbers are
    random, so a re-ingested shard differs in bytes and uploads again.
- Each stage reads a shard's parts through `ingest.part_file`: the copy under `--root` where a
  stage on the box wrote one (all write whole, by a rename, and keep their files after pushing),
  else the Hub's. The box downloads through Xet at ~5 MB/s a stream (plain HTTP, `HF_HUB_DISABLE_XET=1`,
  reached 143 MB/s with 8 at once), and a second copy
  in the HF cache doubles a shard's disk. A part regenerated elsewhere and pushed loses to a stale
  local copy: delete the local one first (`text_en` refilled on another machine would leave voice
  skipping those rows as "rows without text_en" — logged but not voiced). Translation still lists the Hub's segments and translates
  every shard without a local `text_en`: give it `--shards` past the shards translated elsewhere.
  Voice queues a shard only once its `text_en` is on the Hub or here.
- Translate (`training/tranche/translate.py`): Index-Translate-35B-A3B in vLLM on the box, a row
  per prompt, written as `<root>/<source>/text_en/<shard>.parquet` (`id`, `text_en`) and pushed
  with `ingest push <source> --part text_en`; its model choice, checks, speed and runs are in
  [tranche-translate](tranche-translate.md).
- The box's 350 GB disk holds the noisy training cache (143 GB) and a run's checkpoint pair
  (37.5 GB, plus a second while it saves), so tranche data streams through it shard by shard.
  Build uv envs with `--no-cache`: the cache on `/workspace` grew ~35 GB in one env build and
  left 16 GB free, short of the LR test's next checkpoint save (2026-11-04).
- The box's huggingface_hub keeps file contents in a shared store, `$HF_HOME/hub/blobs/<xx>/<sha256>`,
  each with a `.refs` file naming the repo blobs that use it. Deleting a `models--…` or
  `datasets--…` folder frees nothing; delete a shared blob (with its `.lock` and `.refs`) once
  none of its refs exists (32 GB left behind by LongCat and the corpus samples, 2026-11-04).

## Filters

`training/tranche/filters.py` holds what the cache build drops rows on and the text measures behind
it (its Vietnamese-side gates are `filters.vi_gate`, which voice.py also applies, not voicing those
rows); the build applies the thresholds, so they can change without measuring again, except that a
`--max-cer` looser than 0.15 cannot bring back the rows voice skipped at 0.15 (from 2026-11-07)
without voicing them.

- Both filters compare text by `spoken`: lower-case NFC syllables without punctuation, digits
  written out by num2words' Vietnamese, and `ngàn`/`linh`/`nhăm` folded to `nghìn`/`lẻ`/`lăm`.
  PhoAudiobook and GigaSpeech 2 spell numbers; Qwen3-ASR and FLEURS write digits, and on the
  first PhoAudiobook shards rows with digits made 48% of those over 0.1 CER before (15 of 51
  digit rows over 0.1, 4 after). num2words reads a long digit run (a phone number) as one number.
- Evaluation overlap (`overlaps`, with the sets in `data/eval/`; `giga-heldout` applies the same
  rule the other way, [sft-evaluation](sft-evaluation.md#evaluation-sets)): a row is dropped when
  its transcript holds a 13-syllable run of a FLEURS vi dev or test or MultiMed-ST vi select or
  final sentence, or a whole sentence of 9–12 syllables (62 of them). None of the first tranche's
  840,520 rows does (2026-11-05, measured over the text columns alone). Runs of 10 syllables or
  fewer match stock broadcast phrases ("cảm ơn quý vị và các bạn đã quan tâm", 80 rows at 10, 257
  at 8), and the 18 sentences under 9 syllables ("được", "alo", "vùng nông thôn") everyday speech,
  so those are not matched.
- Vietnamese ASR (`<source>/asr_vi/<shard>`: `id`, `asr_vi`, `language`, `cer` against the
  corpus transcript over `spoken`): every row of PhoAudiobook, viVoice and GigaSpeech 2 is heard by
  Qwen3-ASR-1.7B on the shared ASR server, its language detected, not forced, up to 512 tokens;
  YODAS's ingest writes its own ([tranche-yodas](tranche-yodas.md#rows)). The rows where the
  transcript and the ASR disagree, or the language is not Vietnamese, are the ones the build drops.
  - `filters.Hearer` parses the reply: unforced, the model writes `language <Name><asr_text>…`,
    and `language None` for no speech; a reply without that header is all transcript with
    language `""`, as qwen_asr parses it.
  - The client sends 16 kHz WAV: sent as Opus, the server's 8 API processes saturated the box's 27
    cores at 21 rows/s and starved the TTS's main process.
  - First PhoAudiobook shards: CER median 0.000, p90 0.03–0.04, about one row a shard detected
    as another language. The server's batching moves a rerun's transcripts a little: of a shard's
    1,299, 60 differed (48 in punctuation or case only, the rest by at most 0.042 CER).
  - The highest CERs are faults of the corpus: PhoAudiobook transcripts that repeat the clip's
    last clause (0.24–0.34), numbers and punctuation spelled oddly ("chấm ba chấm", "triệu
    bốn nghìn"), abbreviations ("tp. hcm"); then words the ASR loses at a clip's cut edges.
    `spoken` leaves `%` and decimal commas unspoken (the ASR's "80%" against the corpus's "tám
    mươi phần trăm"), which adds at most ~0.04 CER to ~0.5% of rows, so it stays.

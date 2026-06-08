# Vietnamese Fine-Tune Data Plan

Goal: prepare Vietnamese to English data for Hibiki-Zero fine-tuning.

This plan is for later implementation and review. It focuses on data preparation,
not the training loop.

## Summary

Hibiki-Zero speech-to-speech fine-tuning needs paired source speech and target
outputs. For Vietnamese, there is no ready off-the-shelf Vietnamese to English
speech-to-speech corpus, so we build synthetic pairs:

```text
Vietnamese source audio + Vietnamese transcript
  -> English translation text
  -> English TTS audio
  -> cached Mimi source/target audio tokens
```

Start with a 10-hour slice to validate mechanics before scaling.

## Recommended Data Sources

| Role | Dataset | Use |
| --- | --- | --- |
| Primary training source | `linhtran92/viet_bud500` | Vietnamese speech + transcript |
| Optional extra source | Common Voice Vietnamese | More speaker/accent variety |
| Held-out evaluation | NTREX-128 Vietnamese + English | Build an Audio-NTREX-style Vietnamese eval set |

### VietBud500

Use VietBud500 as the primary source. It is a large Vietnamese ASR dataset with
Vietnamese audio and transcripts. Hugging Face access may be gated, so the data
prep implementation must handle authentication and document access requirements.

Expected fields for the raw source data:

```text
audio
transcription
split
```

Exact column names should be confirmed with `load_dataset()` before implementation.

### NTREX Vietnamese Eval

Use NTREX-128 to build a held-out Vietnamese evaluation set similar to
`Audio-NTREX-4L`:

```text
Vietnamese NTREX text -> Vietnamese TTS source audio
English NTREX text    -> target_text reference
```

This eval set should not be mixed into training.

## Target Data Format

The prepared fine-tune manifest should use one row per training chunk.

```csv
sample_id,split,source_audio,source_text,target_text,target_audio,duration_s
vi_000001,train,data/vi_en_10h/source_audio/vi_000001.wav,...,...,data/vi_en_10h/target_audio/en_000001.wav,42.30
```

Required fields:

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable ID for the prepared chunk |
| `split` | `train`, `valid`, or `test` |
| `source_audio` | Vietnamese source speech WAV |
| `source_text` | Vietnamese transcript |
| `target_text` | English translation |
| `target_audio` | English TTS audio synthesized from `target_text` |
| `duration_s` | Source audio duration in seconds |

After Mimi caching, add a cache manifest or metadata file that maps each
`sample_id` to cached token files:

```csv
sample_id,codes_path,num_frames,duration_s
vi_000001,data/vi_en_10h/mimi_cache/vi_000001.npy,529,42.30
```

## Directory Layout

Use a run-specific folder so future experiments are reproducible:

```text
finetune_data/vi_en_10h/
  manifest.csv
  cache_manifest.csv
  metadata.json
  source_audio/
    vi_000001.wav
  target_audio/
    en_000001.wav
  mimi_cache/
    vi_000001.npy
```

`metadata.json` should record:

```json
{
  "source_dataset": "linhtran92/viet_bud500",
  "hours_target": 10,
  "translation_model": "TBD",
  "tts_model": "TBD",
  "sample_rate_hz": 24000,
  "chunk_duration_range_s": [30, 75]
}
```

## Pipeline

### Phase 1: Inspect Source Dataset

- Load VietBud500 in streaming mode.
- Confirm split names, column names, audio format, sample rate, and transcript field.
- Estimate duration distribution and transcript quality.
- Confirm license and gated-access requirements.

Acceptance criteria:

- A small inspection report identifies exact columns and split names.
- At least 10 examples can be loaded without downloading the full dataset.

### Phase 2: Build 10-Hour Vietnamese Source Slice

- Select training rows until source duration reaches about 10 hours.
- Keep a small validation slice separate from training.
- Convert or resample source audio to the sample rate expected by the Hibiki/Mimi path.
- Write `source_audio` files and initial manifest rows.

Acceptance criteria:

- `manifest.csv` contains about 10 hours of source audio.
- No missing source files.
- Durations are recorded correctly.

### Phase 3: Translate Vietnamese Text to English

- Translate `source_text` into English `target_text`.
- Use one translation model consistently for the full run.
- Preserve Vietnamese source text and English target text in the manifest.

Recommended translation options:

| Option | Notes |
| --- | --- |
| NLLB | Open multilingual MT baseline |
| MADLAD | Strong multilingual translation option |
| API-based translation | Higher quality, but less reproducible and may cost money |

Acceptance criteria:

- Every training row has non-empty `target_text`.
- Spot-check at least 50 translations for obvious failures.

### Phase 4: Synthesize English Target Audio

- Generate English `target_audio` from `target_text`.
- Use one TTS voice/configuration for the first 10-hour run unless testing voice diversity.
- Normalize loudness and output WAV files.

Acceptance criteria:

- Every training row has a target audio file.
- Target audio duration is reasonable for the text length.
- No empty, clipped, or corrupt WAV files.

### Phase 5: Chunk Short Clips

If source clips are too short, concatenate consecutive examples into longer chunks.

Target chunk length:

```text
30-75 seconds
```

For each chunk:

- concatenate Vietnamese source audio with small silences between clips
- concatenate `source_text`
- concatenate `target_text`
- synthesize or concatenate matching English target audio

Acceptance criteria:

- Most chunks are within 30-75 seconds.
- Text order matches audio order.
- Chunk boundaries do not mix train and validation data.

### Phase 6: Cache Mimi Codes

Run Mimi encoding offline for each prepared pair:

- encode Vietnamese `source_audio`
- encode English `target_audio`
- save combined token/cache files for training

Acceptance criteria:

- Every manifest row has a cache entry.
- Cache files have expected frame counts for the recorded duration.
- A small sample can be decoded or sanity-checked before training.

## Review Gates

Before training, review:

| Gate | Check |
| --- | --- |
| Dataset access | VietBud500 access works and license is acceptable |
| Manifest integrity | No missing files, empty text, or invalid durations |
| Translation quality | Spot-check translations before TTS |
| Audio quality | Spot-check source and target audio |
| Chunk alignment | Source/target text and audio are in the same order |
| Cache integrity | Mimi cache exists for every row |

## Open Decisions

These must be chosen before implementation:

| Decision | Default |
| --- | --- |
| Translation model | NLLB or MADLAD |
| English TTS provider/model | TBD |
| TTS voice strategy | Single voice for first run |
| Source sample rate | Resample to Hibiki/Mimi expected rate |
| Chunking | 30-75 second chunks |
| First run size | 10 hours |

## Non-Goals

- Do not train the model in this data-prep phase.
- Do not use Audio-NTREX-4L as direct training data unless English target audio
  is synthesized.
- Do not mix held-out NTREX Vietnamese eval data into training.

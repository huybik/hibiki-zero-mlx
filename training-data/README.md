# PhoMT Speech Data Pipeline

This folder builds a paired English/Vietnamese speech dataset from the raw PhoMT text dataset.

Source text dataset:

```text
ura-hcmut/PhoMT
```

Uploaded speech dataset:

```text
anquachdev/PhoMT-en-vi-speech
```

Final uploaded columns:

```text
en, vi, audio_en, audio_vi, duration_en_s, duration_vi_s, duration_ratio_en_vi
```

## Files

- `load_raw.py` loads raw PhoMT text data from Hugging Face.
- `pipeline.py` generates Vietnamese and English audio from PhoMT rows.
- `process.py` is a backward-compatible entry point that runs `pipeline.py`.
- `upload.py` builds the paired audio dataset and pushes it to Hugging Face.
- `load_train_data.py` loads the uploaded speech dataset for training or coworker preview.

Generated audio, local cache, and preview folders are written outside the repo under:

```text
D:\Code\datasets
```

## Setup

Run commands from the repo root:

```powershell
uv sync
```

For CUDA generation, make sure the environment has a CUDA-enabled PyTorch build. The current project is configured for:

```text
torch==2.9.1
PyTorch CUDA index: https://download.pytorch.org/whl/cu128
```

## Preview Raw PhoMT

```powershell
uv run python training-data/load_raw.py
```

This prints a small preview from the raw PhoMT train split.

## Generate Speech

Edit the config at the top of `pipeline.py`:

```python
START_INDEX = 0
N_SAMPLES = 128
BATCH_SIZE = 8
LANGUAGES = ("vi","en")
TTS_DEVICE = "cuda"
PARALLEL_LANGUAGES = True
```

Then run:

```powershell
uv run python training-data/pipeline.py
```

When `PARALLEL_LANGUAGES = True`, the parent command launches separate child processes:

```text
vi worker -> D:\Code\datasets\vieNeu\outputs\vi\
en worker -> D:\Code\datasets\english\outputs\en\
```

Each output folder contains WAV files and a `manifest.csv`.

Kokoro `af_nicole` is generated faster than the default because it is otherwise
much slower than the paired Vietnamese speech.

If GPU memory is tight, set:

```python
PARALLEL_LANGUAGES = False
```

## Upload Dataset

After both Vietnamese and English audio are generated for matching indexes:

```powershell
uv run python training-data/upload.py
```

The uploader reads manifests from `D:\Code\datasets\vieNeu\outputs\vi\` and
`D:\Code\datasets\english\outputs\en\`.

This builds rows with:

```text
en, vi, audio_en, audio_vi, duration_en_s, duration_vi_s, duration_ratio_en_vi
```

Rows with EN/VI duration ratio outside the configured range in `upload.py` are
skipped before upload.

and pushes to:

```text
anquachdev/PhoMT-en-vi-speech
```

## Load Uploaded Dataset

For training or coworker preview:

```powershell
uv run python training-data/load_train_data.py
```

The loader keeps audio as raw file/bytes by default with:

```python
Audio(decode=False)
```

This avoids requiring `torchcodec` just to inspect or copy audio samples. It writes a few preview WAV files to:

```text
D:\Code\datasets\audios\
```

## Notes

- Vietnamese TTS uses VieNeu.
- English TTS uses Kokoro-82M.
- Voice gender is matched per sample when `MATCH_VOICE_GENDER = True`.
- The generated audio estimate from the current sample is roughly 11.4k total hours for the full PhoMT train split if both languages are synthesized.

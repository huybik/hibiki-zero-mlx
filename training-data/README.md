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

- `paths.py` shared data locations (override the data root with `PHOMT_DATA_DIR`).
- `load_raw.py` loads raw PhoMT text data from Hugging Face.
- `pipeline.py` generates Vietnamese and English audio from PhoMT rows.
- `upload.py` builds the paired audio dataset and pushes it to Hugging Face.
- `load_train_data.py` loads the uploaded speech dataset for training or coworker preview.

Generated audio, local cache, and preview folders are written outside the repo under `PHOMT_DATA_DIR` (default: `D:\Code\datasets` on Windows, `~/datasets` elsewhere).

## Setup

The pipeline runs in its own conda env `phomt-data` (torch 2.13 — its MPS ops run as native Metal kernels, a free speedup on Apple Silicon):

```bash
conda create -y -n phomt-data python=3.12
conda run -n phomt-data pip install "torch==2.13.*" "kokoro>=0.9.2" "vieneu>=3.0.9" \
    datasets soundfile "onnxruntime<1.24" hf-xet coremltools \
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# coremltools powers the Kokoro Core ML/Metal GPU decoder (kokoro_coreml.py);
# macOS-only, harmless to skip on CUDA boxes (set EN_COREML_WORKERS = 0).
# torchaudio ended at 2.11 (maintenance mode); VieNeu only needs its pure-Python
# kaldi fbank, which works fine against torch 2.13 — install without deps so
# torch is not downgraded.
conda run -n phomt-data pip install --no-deps torchaudio==2.11.0
conda activate phomt-data
```

On CUDA machines install the matching CUDA wheel of torch instead (`--index-url https://download.pytorch.org/whl/cu126`).

Device selection is automatic (`TTS_DEVICE = "auto"` in `pipeline.py`): cuda > mps > cpu. On Apple Silicon the pipeline runs on MPS with VieNeu's PyTorch backend; vieneu >= 3.2 batches natively on both CUDA and MPS (measured on M4 Pro: batch 8 -> 6.1x real-time vs 3.1x sequential).

## Voice Bank (VI speaker diversity)

```bash
python training-data/build_voice_bank.py
```

Downloads VIVOS (46 Vietnamese speakers, CC BY-NC-SA), picks one 6-12 s reference
clip per speaker, enrolls each with VieNeu voice cloning, and saves the bank to
`<PHOMT_DATA_DIR>/voice_bank/vi_voices.json`. When that file exists, `pipeline.py`
automatically registers the cloned voices and merges them (with genders) into the
VI voice pool — 48 voices instead of the 14 built-in presets. Delete the JSON to
revert to presets only.

## Preview Raw PhoMT

```bash
python training-data/load_raw.py
```

This prints a small preview from the raw PhoMT train split.

## Generate Speech

Edit the config at the top of `pipeline.py`:

```python
START_INDEX = 0
N_SAMPLES = 128
BATCH_SIZE = 8
LANGUAGES = ("vi","en")
TTS_DEVICE = "auto"
PARALLEL_LANGUAGES = True
```

Then run:

```bash
python training-data/pipeline.py
```

When `PARALLEL_LANGUAGES = True`, the parent command launches separate child processes:

```text
vi workers -> <PHOMT_DATA_DIR>/vieNeu/outputs/vi/
en workers -> <PHOMT_DATA_DIR>/english/outputs/en/
```

Each output folder contains WAV files and a `manifest.csv`.

Kokoro `af_nicole` is generated faster than the default because it is otherwise
much slower than the paired Vietnamese speech.

If GPU memory is tight (each worker loads its own model copy — on a Mac start
with 1 worker per language), set:

```python
PARALLEL_LANGUAGES = False   # or VI_WORKERS = 1, EN_WORKERS = 1
```

## Upload Dataset

After both Vietnamese and English audio are generated for matching indexes:

```bash
python training-data/upload.py
```

The uploader reads manifests from `<PHOMT_DATA_DIR>/vieNeu/outputs/vi/` and
`<PHOMT_DATA_DIR>/english/outputs/en/`.

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

## Storage-Saving Resume Workflow

Use this loop when local disk space is limited and the previous batches are
already uploaded to Hugging Face.

1. Pick the next batch range in `training-data/pipeline.py`:

```python
START_INDEX = 20000
N_SAMPLES = 1000
```

Use a `START_INDEX` that is after the rows already generated or uploaded. Keep
`N_SAMPLES` small enough that the generated WAV files fit on local disk.

2. Generate only that batch:

```bash
python training-data/pipeline.py
```

This creates the current batch under:

```text
<PHOMT_DATA_DIR>/vieNeu/outputs/vi/
<PHOMT_DATA_DIR>/english/outputs/en/
```

Each folder must contain its current batch WAV files and `manifest.csv`.

3. Upload the current batch immediately:

```bash
python training-data/upload.py
```

With `RESUME_UPLOAD = True`, `upload.py` checks the existing Hugging Face
dataset, skips already-uploaded `(en, vi)` text pairs, and appends only new
rows. Existing remote Parquet files are left untouched.

4. Confirm the command printed a successful append or push message.

Examples:

```text
Appended 1000 rows in 1 shard(s) to https://huggingface.co/datasets/anquachdev/PhoMT-en-vi-speech
Pushed to https://huggingface.co/datasets/anquachdev/PhoMT-en-vi-speech
```

5. After upload success, delete local generated data for the finished batch:

```text
<PHOMT_DATA_DIR>/vieNeu/outputs/vi/
<PHOMT_DATA_DIR>/english/outputs/en/
<PHOMT_DATA_DIR>/phomt-en-vi-speech/
<PHOMT_DATA_DIR>/.hf_cache/
```

Do not delete the current batch's WAV files or manifests before upload finishes.
The uploader needs the local files while it builds the Parquet shard.

6. Repeat with the next `START_INDEX`.

The safe loop is:

```text
generate one batch -> upload that batch -> delete local batch -> generate next batch
```

The Hugging Face dataset is the permanent copy. `<PHOMT_DATA_DIR>` can be
treated as temporary working storage for the next batch once the upload has
succeeded.

Important: resume detection uses the uploaded `(en, vi)` text pair. If the same
text pair is regenerated with new audio, `upload.py` skips it instead of
replacing the existing Hugging Face row.

## Load Uploaded Dataset

For training or coworker preview:

```bash
python training-data/load_train_data.py
```

The loader keeps audio as raw file/bytes by default with:

```python
Audio(decode=False)
```

This avoids requiring `torchcodec` just to inspect or copy audio samples. It writes a few preview WAV files to:

```text
<PHOMT_DATA_DIR>/audios/
```

## Notes

- Vietnamese TTS uses VieNeu.
- English TTS uses Kokoro-82M.
- Voice gender is matched per sample when `MATCH_VOICE_GENDER = True`.
- The generated audio estimate from the current sample is roughly 11.4k total hours for the full PhoMT train split if both languages are synthesized.

# Hibiki MLX

Hibiki-Zero speech translation with two maintained paths:

- MLX inference on Apple Silicon (upstream q4 for FR/ES/PT/DE → English, or staged BF16
  checkpoints);
- full-model PyTorch SFT on one CUDA H100 for Vietnamese → English.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero) ·
[paper](https://arxiv.org/abs/2602.11072) · weights license CC BY-NC-SA 4.0.

## Layout

```text
inference/            Apple Silicon · MLX
  main.py             file and microphone translation CLI
  hibiki_mlx/         pipelined runtime (CPU Mimi codec overlapped with the GPU LM)
  moshi-mlx/          minimal vendored MLX model with the Hibiki-Zero deltas
  tools/              q4/BF16 conversion, quality gate, benchmark, Swift check
  assets/leon.wav     French gate clip
  macos/              native SwiftUI file and microphone test app
training/             CUDA · PyTorch
  h100.sh             pinned H100 setup, preflight, train and resume with HF recovery
  paths.py            repository defaults and file guards
  prep/               en_speech corpus builder, pair rows → grounded Mimi + CTC caches
  finetune/           full-model SFT trainer, evaluation, HF recovery sync
docs/                 plans, analyses and the technical report
weights → …           upstream and staged models (gitignored, shared by both paths)
data/  runs/          caches and training runs (gitignored)
translations/         inference outputs (gitignored)
```

## Inference

```bash
pip install -e ./inference/moshi-mlx
pip install -e .
```

Download and convert the 3B checkpoint:

```bash
hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
python inference/tools/convert_mlx_q4.py
```

Run file or microphone translation:

```bash
python inference/main.py inference/assets/leon.wav
python inference/main.py --mic
```

`--model` accepts `3b` or a staged q4/BF16 Hibiki-Zero model directory.

Build and open the native macOS test app:

```bash
./inference/macos/HibikiTestApp/build.sh
open "inference/macos/HibikiTestApp/build/Hibiki Test.app"
```

The SwiftUI app provides model and audio pickers, file translation and playback,
live microphone start/stop, the English transcript, and the backend log. It runs
`inference/main.py` with the conda Python at `/opt/homebrew/Caskroom/miniconda/base/bin/python`.

The inference tools are:

- `inference/tools/convert_mlx_q4.py`: convert the 3B PyTorch LM to MLX q4.
- `inference/tools/convert_mlx_bf16.py`: stage an exact PyTorch checkpoint as MLX BF16.
- `inference/tools/verify_mlx_q4.py`: translate the gate clip as a quality check.
- `inference/tools/bench.py`: stage timing and silence-input gate.
- `inference/tools/check_swift_compat.py`: strict group-size-32 artifact validation.

## Training

Training runs only on a Linux pod with one H100, through the pinned CUDA 12.8 launcher:

```bash
./training/h100.sh setup
./training/h100.sh preflight
./training/h100.sh train RUN_NAME --cache-dir data/cache/EN_SPEECH_TRAIN data/cache/phomt-grounded-v2-5s --val-cache-dir data/cache/VALIDATION
```

Training modules run from the repository root as `python -m training.<package>.<module>`.
The only trainer is `training.finetune.train`: base-start, full-model SFT with fp32 master
weights and bf16 autocast on grounded caches. Checkpoints are exact full-model states;
partial or adapter checkpoints are rejected.

The training data is the 5–22.4 s clips of the English read-speech corpus `en_speech`
(Common Voice 27.0 EN + Libriheavy, built by `training.prep.build_en_speech`) plus the
~0.8k-hour PhoMT grounded cache, together ~11.3k hours. See the
[pod guide](docs/finetune.md) and [data preparation](training/prep/README.md).

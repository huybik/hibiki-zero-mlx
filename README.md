# Hibiki MLX

Hibiki-Zero speech translation with two maintained paths:

- q4 MLX inference on Apple Silicon for FR/ES/PT/DE → English;
- full-model PyTorch SFT on CUDA for Vietnamese → English.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero) ·
[paper](https://arxiv.org/abs/2602.11072) · weights license CC BY-NC-SA 4.0.

## Inference

```bash
pip install -e ./moshi-mlx
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
python scripts/convert_mlx_q4.py
```

Run file or microphone translation:

```bash
python main.py assets/samples/leon.wav
python main.py --mic
```

Build and open the native macOS test app:

```bash
./macos/HibikiTestApp/build.sh
open "macos/HibikiTestApp/build/Hibiki Test.app"
```

The SwiftUI app provides model and audio pickers, file translation and playback,
live microphone start/stop, the English transcript, and the backend log. It runs
`main.py` with the conda Python at `/opt/homebrew/Caskroom/miniconda/base/bin/python`.

`main.py` and `hibiki_mlx.pipeline` overlap the CPU Mimi encoder/decoder with the
GPU language model. `--model` accepts `3b` or a staged q4/BF16 Hibiki-Zero model
directory.

The maintained inference utilities are:

- `scripts/convert_mlx_q4.py`: convert the 3B PyTorch LM to MLX q4.
- `scripts/convert_mlx_bf16.py`: stage an exact BF16 checkpoint for MLX listening.
- `scripts/verify_mlx_q4.py`: translate the checked-in sample as a quality gate.
- `scripts/bench.py`: stage timing and silence-input gate.
- `scripts/check_swift_compat.py`: strict group-size-32 artifact validation.

## Training

For an H100 pod, use the checked and pinned CUDA 13.2 workflow:

```bash
./finetune/h100.sh setup
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

For other CUDA development environments, install PyTorch followed by the
training dependencies:

```bash
pip install -e '.[training]'
pip install --no-deps moshi==0.2.13
```

Install the CUDA PyTorch build first. `--no-deps` prevents Moshi's stale Torch
constraint from replacing it.

The only supported trainer is `finetune/train.py`: base-start, full-model SFT
with fp32 master weights and CUDA bf16 autocast. Checkpoints are exact full-model
states; partial or adapter checkpoints are rejected.

See [SFT mechanics](docs/finetune.md), the current
[training recipe](docs/training_plan.md), and the
[validation contract](docs/validation_plan.md).

## Layout

- `hibiki_mlx/`: pipelined q4 inference runtime.
- `moshi-mlx/`: minimal vendored MLX model implementation with Hibiki deltas.
- `finetune/`: full-model SFT, cache preparation, and evaluation.
- `remote_dataset/`: reproducible FLEURS downloader.
- `macos/HibikiTestApp/`: native SwiftUI file and microphone test app.
- `scripts/`: q4 conversion and inference verification.
- `assets/samples/`: the retained inference gate clip.

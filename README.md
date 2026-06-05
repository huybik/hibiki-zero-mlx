# Hibiki-Zero

Hibiki-Zero is a real-time and multilingual speech translation model.
It translates from French, Spanish, Portuguese and German to English: accurately, with low latency, high audio quality, and voice transfer.

https://github.com/user-attachments/assets/d533ec45-8d5e-4e41-886a-0b2d198be6f3

[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) | 
[⚙️ Tech report](https://kyutai.org/blog/2026-02-12-hibiki-zero) |
[📄 Paper](https://arxiv.org/abs/2602.11072) |
[🎧 More samples](https://huggingface.co/spaces/kyutai/hibiki-zero-samples)

## Requirements

Hibiki-Zero is a 3B-parameter model. It runs on:
- an **NVIDIA GPU** (8 GB VRAM should work, 12 GB is safe), or
- **Apple Silicon** — this fork patches the model to run on Mac via PyTorch/MPS or natively in MLX (see [Apple Silicon](#apple-silicon-mps--mlx) below).

## Run the server

Hibiki-Zero comes with a server you can run to interact with Hibiki in real time. To run it, just use:

```python
uvx -p 3.13 hibiki-zero serve [--gradio-tunnel]
```

Then go to the URL displayed to try out Hibiki-Zero.
The `--gradio-tunnel` flag will forward the server to a public URL that you can access from anywhere.

If you don't have `uv`, you must first install hibiki-zero with `pip install hibiki-zero` and then run the server with `hibiki-zero serve [--gradio-tunnel]`.

## Run inference

If you'd like to run Hibiki-Zero on existing audio files, run:

```python
uvx -p 3.13 hibiki-zero generate [--file /path/to/my/audio.wav --file /path/to/another/audio.mp3]
```

Batch inference is supported, meaning you can run the model on multiple audio files at the same time.

## Apple Silicon (MPS / MLX)

This fork adds two ways to run Hibiki-Zero on a Mac (the upstream code is NVIDIA-only). Both expect the checkpoint files in `weights/` (`config.json`, the `hibiki-*` and `mimi-*` safetensors, and the tokenizer).

### Download the weights

Pull the checkpoint from Hugging Face into `weights/` (~6.2 GB):

```bash
pip install -U "huggingface_hub[cli]"
hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
```

The model is gated (CC BY-NC-SA 4.0) — accept the terms on the [model page](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) and run `hf auth login` first if the download 401s.

### PyTorch / MPS

Pass `--device mps`; the CUDA-only guards are patched out so `serve` and `generate` work on Apple GPUs. Runs at roughly **0.7× real-time** — fine for offline batch translation.

```bash
hibiki-zero generate --device mps \
  --config-path weights/config.json \
  --model-weight "weights/hibiki-pytorch-77f82164@110.safetensors" \
  --mimi-weight "weights/mimi-pytorch-e351c8d8@125.safetensors" \
  --tokenizer weights/tokenizer_spm_48k_multi6_2.model \
  --file hibiki_zero/samples/leon.wav --gen-duration 66 --out-dir translations
```

### MLX 4-bit (faster)

Native MLX inference via [`moshi-mlx`](https://pypi.org/project/moshi-mlx/), quantized to 4-bit. The LM shrinks from **5.8 GB → 2.2 GB** and runs at **~1.3× real-time** (faster than the bf16 MPS path). `mlx_hibiki_patch.py` adds the hibiki-zero deltas that stock `moshi-mlx` misses (`hidden_scale`, grouped-query attention / `kv_repeat=2`, and `rope_concat`).

```bash
pip install moshi-mlx
python convert_mlx_q4.py   # writes weights/hibiki.q4.safetensors
python verify_mlx_q4.py    # translates samples/leon.wav -> translations/leon_mlx_q4.wav
```

## Local development

We recomment using `uv`, run anything with `uv run` in this repository. For example

```bash
uv run some_file.py
or 
uv run hibiki-zero serve
```
if you use pip, use `pip install -e .` before executing python commands.


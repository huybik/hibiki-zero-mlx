# Hibiki-Zero MLX

Apple Silicon port of **Hibiki-Zero** — runs the model on Mac via PyTorch/MPS or natively in MLX (4-bit). Derived from [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero), now an independent repo.

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
- **Apple Silicon** — this repo patches the model to run on Mac via PyTorch/MPS or natively in MLX (see [Apple Silicon](#apple-silicon-mps--mlx) below).

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

This repo adds two ways to run Hibiki-Zero on a Mac (the upstream code is NVIDIA-only). Both expect the checkpoint files in `weights/` (`config.json`, the `hibiki-*` and `mimi-*` safetensors, and the tokenizer).

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

Native MLX inference via [`moshi-mlx`](https://pypi.org/project/moshi-mlx/), quantized to 4-bit. The LM shrinks from **5.8 GB → 2.2 GB** and runs at **~1.3× real-time** (faster than the bf16 MPS path). `src/mlx_hibiki_patch.py` adds the hibiki-zero deltas that stock `moshi-mlx` misses (`hidden_scale`, grouped-query attention / `kv_repeat=2`, and `rope_concat`).

Run from the repo root (`main.py`/scripts add `src/` to the path and anchor on it):

```bash
pip install moshi-mlx sounddevice
python main.py samples/leon.wav         # file  -> translations/leon_translated.wav (~3x RT)
python main.py --mic                    # realtime mic -> speakers (Ctrl-C to stop)
```

`main.py` is the entry point for both. To (re)build the q4 weights or verify them:

```bash
python scripts/convert_mlx_q4.py   # writes weights/hibiki.q4.safetensors
python scripts/verify_mlx_q4.py    # translates samples/leon.wav -> translations/leon_mlx_q4.wav (pipelined, ~3x RT)
```

Pre-quantized weights (no need to run `scripts/convert_mlx_q4.py`) are published at
[`huybik/hibiki-zero-3b-mlx-q4`](https://huggingface.co/huybik/hibiki-zero-3b-mlx-q4).

#### Results (Apple M4 Pro, `samples/leon.wav`, FR→EN)

|              | Value |
|--------------|-------|
| **LM weights** | 5.8 GB bf16 → **2.2 GB** q4 (578 layers quantized) |
| **Speed**    | 16.5 tok/s ≈ **1.3× real-time** — vs ~0.7× for the PyTorch/MPS path (~1.9× faster) |
| **Quality**  | Coherent FR→EN — correctly translated the Léon Marchand / Paris 2024 Olympics commentary; minor q4 artifacts (e.g. one "Paris 1024" slip) |
| **Audio out** | `translations/leon_mlx_q4.wav` (2.9 MB, 24 kHz) ✅ |

The q4 weights use `group_size=32`; this is required because stock `moshi-mlx`
(and the moshi-swift iOS loader) hardcode gs32 for `.q4.safetensors`.

### Pipelined MLX (faster still — overlaps codec with the LM)

Profiling the q4 decode loop showed the Mimi codec (`rustymimi`, CPU) was **~58%**
of each frame, run strictly *sequentially* with the GPU LM — CPU and GPU idling on
each other. `src/infer_mlx_fast.py` overlaps them: an **encoder thread** streams the
whole file ahead, the **main thread** runs only the GPU LM step, and a **decoder
thread** turns the audio tokens back into PCM. FIFO queues preserve streaming order,
so the output is identical to the sequential path — `rustymimi` releases the GIL, so
the threads run truly concurrently (each thread gets its own `Tokenizer` instance).

```bash
python scripts/verify_mlx_q4.py [in.wav] [out.wav]   # defaults: samples/leon.wav -> translations/leon_mlx_q4.wav
```

`verify_mlx_q4.py` now uses this path under the hood, so the MLX entry point is fast
by default. (The PyTorch `serve`/`generate` commands are a separate stack — codec on
the MPS GPU, live-streaming — so this CPU/GPU-overlap trick doesn't transfer there.)

For an offline file the encode stream runs entirely ahead and decode hides under the
GPU, so per-frame wall collapses to ~the LM cost alone:

|              | frames/s | × real-time | ms/frame |
|--------------|----------|-------------|----------|
| Sequential (`run_inference`) | 16.9 | 1.35× | 59 |
| **Pipelined** (`src/infer_mlx_fast.py`) | **37.5** | **3.0×** | **27** |

(Apple M4 Pro, `samples/leon.wav`, FR→EN, q4 — **2.2× faster**, output byte-identical.)
`scripts/profile_mlx.py` prints the per-stage breakdown (mimi encode / LM main / LM depformer
/ mimi decode) used to find this.

## Local development

We recomment using `uv`, run anything with `uv run` in this repository. For example

```bash
uv run some_file.py
or 
uv run hibiki-zero serve
```
if you use pip, use `pip install -e .` before executing python commands.


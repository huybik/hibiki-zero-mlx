# Hibiki MLX

Realtime speech-to-speech translation on Apple Silicon. This repo is an MLX runtime for
Kyutai's **Hibiki-Zero** (FR/ES/PT/DE → EN simultaneous translation, 3B LM + Mimi codec
@ 12.5 Hz), quantized to 4-bit and pipelined to **~3× real-time on an M4 Pro**. The goal is a
runtime that fits the **iPhone 80 ms/frame budget** — see [docs/refactor_plan.md](docs/refactor_plan.md)
and [vision/vision.md](vision/vision.md).

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero) ·
[paper](https://arxiv.org/abs/2602.11072) · weights license CC BY-NC-SA 4.0.

## Setup

```bash
pip install -e ./moshi-mlx   # vendored moshi-mlx fork with the hibiki-zero deltas
pip install -e .             # the hibiki_mlx runtime package
```

Download the checkpoint into `weights/` (~6.2 GB, gated — accept terms on the
[model page](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) and `hf auth login`):

```bash
hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
python scripts/convert_mlx_q4.py   # -> weights/hibiki.q4.safetensors (2.2 GB, q4 gs32)
```

Pre-quantized weights: [`huybik/hibiki-zero-3b-mlx-q4`](https://huggingface.co/huybik/hibiki-zero-3b-mlx-q4)
(3B) and [`huybik/hibiki-1b-mlx-q4`](https://huggingface.co/huybik/hibiki-1b-mlx-q4) (Hibiki-M 1B,
staged by `scripts/convert_hibiki_m_mlx_q4.py`).

## Run

```bash
python main.py assets/samples/leon.wav   # file -> translations/leon_translated.wav + .txt (~3x RT)
python main.py --mic                     # realtime mic -> speakers (needs sounddevice; Ctrl-C stops)
```

Both modes run the q4 weights through `hibiki_mlx.pipeline` (`load()`/`run()`): the CPU Mimi
codec (rustymimi, GIL-free) is pipelined across encoder/decoder threads so the critical path is
just the GPU LM step (~24 ms/frame on M4 Pro). Library use:

```python
from hibiki_mlx import load, run
run("audio.wav", "out.wav")   # writes wav + text sidecar
```

## Scripts

- `scripts/verify_mlx_q4.py` — gate: translates `assets/samples/leon.wav` → `translations/leon_mlx_q4.wav`.
- `scripts/profile_mlx.py` — per-stage frame profile (mimi encode / LM main / depformer / decode).
- `scripts/convert_mlx_q4.py`, `scripts/convert_hibiki_m_mlx_q4.py` — q4 conversion (keep `group_size=32`,
  required by stock moshi-mlx/moshi-swift loaders).
- `scripts/push_mlx_q4.py`, `scripts/push_hibiki_m_mlx_q4.py` — publish weights to HF.

## Repo map

- `hibiki_mlx/` — the runtime package (pipelined q4 inference).
- `moshi-mlx/` — vendored moshi-mlx fork; owns the hibiki-zero model deltas (GQA, `hidden_scale`,
  `rope_concat`, depformer output LayerNorm).
- `finetune/` — PyTorch LoRA training stack for Vietnamese adaptation (FLEURS), see `docs/finetune.md`.
- `remote_dataset/` — CoVoST2 / FLEURS eval datasets + batch translate/score scripts.
- `training-data/` — PhoMT synthetic TTS data pipeline.
- `docs/` — [refactor plan](docs/refactor_plan.md), [distill plan](docs/distill_plan.md),
  [training explainer](docs/training_explainer.md), [report](docs/report.md).

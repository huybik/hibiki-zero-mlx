# Hibiki-Zero — local test project

Running Kyutai's **Hibiki-Zero** (simultaneous speech-to-speech + speech-to-text translation, FR/ES/PT/DE → EN) locally on an **Apple M4 Pro via MPS**. Upstream is NVIDIA-only; this project patches it to run on Mac.

- Model: `kyutai/hibiki-zero-3b-pytorch-bf16` (3B hierarchical transformer + Mimi codec @ 12.5 Hz). License CC BY-NC-SA 4.0.
- Repo/paper: https://github.com/kyutai-labs/hibiki-zero · https://arxiv.org/abs/2602.11072v1

## Layout (`/Volumes/data/models/hibiki-zero/`)
- `code/` — cloned inference repo (`hibiki_zero/run.py` = CLI, `inference.py` = core). Patched + installed editable.
- `code/weights/` — the checkpoint (gitignored): `config.json`, `hibiki-pytorch-77f82164@110.safetensors` (5.8 GB), `mimi-pytorch-e351c8d8@125.safetensors` (367 MB), `tokenizer_spm_48k_multi6_2.model`.
- `code/hibiki_zero/static/` — built Next.js frontend (from `build_frontend.sh`, needed by `serve`).
- `.venv/` — isolated venv (py3.13): hibiki-zero 0.0.4, moshi 0.2.13, moshi-mlx 0.3.0, torch 2.9.1. **Never** install into the conda base env.
- `code/translations/` — `generate` outputs (mono + stereo wav + txt), gitignored.
- `code/hibiki_zero/samples/` — bundled test clips: `leon.wav` (~64 s FR Olympics), `crepes.mp3`.

## MPS patches (in `code/`, why it runs on Mac)
1. `run.py`: CUDA guard → `if device == "cuda" and not torch.cuda.is_available()` (both `serve` and `generate`).
2. `inference.py`: `torch.cuda.synchronize()` wrapped in `if torch.cuda.is_available()`.
Installed with `pip install -e`, so edits stay live.

## Run
Batch translate a file (weights live in `code/weights/`):
```bash
cd /Volumes/data/models/hibiki-zero
./.venv/bin/hibiki-zero generate --device mps \
  --config-path code/weights/config.json \
  --model-weight "code/weights/hibiki-pytorch-77f82164@110.safetensors" \
  --mimi-weight "code/weights/mimi-pytorch-e351c8d8@125.safetensors" \
  --tokenizer code/weights/tokenizer_spm_48k_multi6_2.model \
  --file code/hibiki_zero/samples/leon.wav --gen-duration 66 --out-dir code/translations
```
Web UI (same flags, `serve` instead of `generate`): http://localhost:8998

## MLX 4-bit path (faster alternative to PyTorch/MPS)
Native MLX inference via `moshi-mlx` 0.3.0 (installed in `.venv`). **~1.3× real-time** (16.5 tok/s @ 12.5 Hz) vs ~0.7× for the bf16 MPS path; LM weights 5.8 GB → **2.2 GB** q4.
- `code/mlx_hibiki_patch.py` — runtime patches `moshi_mlx` for hibiki-zero (stock pkg targets moshi/older-hibiki): honours `hidden_scale=6` + `kv_repeat=2` in config, adds grouped-query attention to the forward pass, and wires `rope_concat` (RoPE interleave=False). Import before building/loading the model.
- `code/convert_mlx_q4.py` — one-shot: load PyTorch LM → `nn.quantize(bits=4, group_size=32)` → `weights/hibiki.q4.safetensors`. Mimi codec stays separate/bf16.
- `code/verify_mlx_q4.py` — translates `leon.wav` via patched `moshi_mlx.run_inference` (uses `rustymimi` for the codec; our mimi sig `e351c8d8` loads directly). Output: `translations/leon_mlx_q4.wav`. Verified coherent FR→EN.
- Published q4 weights + patch + model card: [`huybik/hibiki-zero-3b-mlx-q4`](https://huggingface.co/huybik/hibiki-zero-3b-mlx-q4). **Keep `group_size=32`** — stock `moshi-mlx`/moshi-swift hardcode gs32 for `.q4.safetensors` (`run_inference.py:80`); gs64 saves ~240 MB but won't load without patching every loader (bad for the iOS/moshi-swift path).

## Notes / gotchas
- Input must be FR/ES/PT/DE and **≤ `--gen-duration`** seconds (max 120). Every input is padded to gen-duration, so smaller = faster.
- Default dtype float16 (no MPS op errors); `--bf16` available.
- Performance: ~0.7× real-time on MPS (H100 does ~3× faster than real-time) — fine for offline, laggy for live.
- Verified working: bundled `samples/leon.wav` (~64 s FR Olympics) → coherent EN text + 24 kHz audio (mono + stereo wav).

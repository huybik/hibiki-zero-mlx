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
- `code/mlx_hibiki_patch.py` — runtime patches `moshi_mlx` for hibiki-zero (stock pkg targets moshi/older-hibiki): (1) honours `hidden_scale=6` + `kv_repeat=2` in config; (2) adds grouped-query attention to the forward pass; (3) wires `rope_concat` (RoPE interleave=False); (4) adds the learned per-slice depformer output **LayerNorm** (`depformer_norms.{i}`, applied before each audio `linear_out`) that moshi_mlx omits. Import before building/loading the model. **Without (4) the audio babbles** — overlapping/interleaved voices + clipping (depformer logits come out ~3× too small → out-of-distribution audio tokens); the text stream is unaffected since it never passes through the depformer. Diagnose with a silence-in test: feed zeros, a correct model stays near-silent (rms≈0.06, peak<1.0); the bug gives rms≈0.13, peak≈1.23 (clipping). **q4 weights must be regenerated** (`convert_mlx_q4.py`) after touching the depformer — the LayerNorm stays bf16 (nn.quantize skips non-Linear).
- `code/convert_mlx_q4.py` — one-shot: load PyTorch LM → `nn.quantize(bits=4, group_size=32)` → `weights/hibiki.q4.safetensors`. Mimi codec stays separate/bf16.
- `code/verify_mlx_q4.py` — translates `leon.wav` via patched `moshi_mlx.run_inference` (uses `rustymimi` for the codec; our mimi sig `e351c8d8` loads directly). Output: `translations/leon_mlx_q4.wav`. Verified coherent FR→EN.
- Published q4 weights + patch + model card: [`huybik/hibiki-zero-3b-mlx-q4`](https://huggingface.co/huybik/hibiki-zero-3b-mlx-q4). **Keep `group_size=32`** — stock `moshi-mlx`/moshi-swift hardcode gs32 for `.q4.safetensors` (`run_inference.py:80`); gs64 saves ~240 MB but won't load without patching every loader (bad for the iOS/moshi-swift path).

## Speed work (in progress)
Profiled q4 decode (`code/profile_mlx.py`, per-stage w/ mx.eval barriers): mimi codec (rustymimi, **CPU**) was ~58%/frame (enc 30% + dec 28%), run *sequentially* with the GPU LM (main 20% + depformer 21%). rustymimi **releases the GIL**.
- `code/infer_mlx_fast.py` — overlaps codec with LM via 3 threads (encoder runs whole file ahead → queue → main-thread GPU LM step → decoder thread). Each thread needs its **own** `rustymimi.Tokenizer` (one instance borrowed by 2 threads → "Already borrowed" panic). FIFO queues keep streaming order ⇒ output byte-identical. **1.35× → 3.0× RT** (16.9 → 37.5 frames/s, 2.2×), zero quality change. Codec now hidden; GPU LM (~25 ms/frame) is the new floor. Reusable `load()`/`run()`.
- `code/verify_mlx_q4.py` — now wired to `infer_mlx_fast.run()` (the MLX entry point; ~2.8–3× RT). The PyTorch `serve`/`generate` in `run.py` are a **separate stack** (codec on MPS GPU, live-streaming) — this CPU/GPU-overlap trick doesn't apply there.
- Next: `mx.compile` the LM step (depformer loop) to shave the now-exposed ~25 ms → target ~4–5× RT.

## Notes / gotchas
- Input must be FR/ES/PT/DE and **≤ `--gen-duration`** seconds (max 120). Every input is padded to gen-duration, so smaller = faster.
- Default dtype float16 (no MPS op errors); `--bf16` available.
- Performance: ~0.7× real-time on MPS (H100 does ~3× faster than real-time) — fine for offline, laggy for live.
- Verified working: bundled `samples/leon.wav` (~64 s FR Olympics) + `crepes.mp3` (~56 s FR recipe) → coherent EN text + 24 kHz audio (mono + stereo wav), on both the PyTorch/MPS path (`0_*` outputs) and the fixed MLX q4 path (`*_mlx_fixed.wav`, ~3× RT, no babble/clipping after the depformer-LayerNorm fix).

# Hibiki MLX — project context

MLX runtime for Kyutai's **Hibiki-Zero** (simultaneous speech-to-speech + speech-to-text translation, FR/ES/PT/DE → EN, 3B hierarchical transformer + Mimi codec @ 12.5 Hz) on Apple Silicon, aimed at the **iPhone 80 ms/frame budget**. The upstream PyTorch stack was purged; the MLX q4 path is the product. **Phase 1 of `docs/refactor_plan.md` (repo restructure & dead-code purge) is done.**

- Upstream: https://github.com/kyutai-labs/hibiki-zero · https://arxiv.org/abs/2602.11072v1 · weights CC BY-NC-SA 4.0.
- Python env: conda base at `/opt/homebrew/Caskroom/miniconda/base/bin/python` (py3.13; mlx 0.31, torch 2.12, moshi 0.2.13, rustymimi 0.4.1; `moshi-mlx/` and `hibiki_mlx` installed editable).

## Layout (repo root = `code/`)
- `hibiki_mlx/` — THE runtime package (installed editable). `pipeline.py` = pipelined q4 inference `load()`/`run()`/`make_mimi()`; re-exported from `hibiki_mlx`. No `sys.path` hacks anywhere; weights anchor on repo root (`hibiki_mlx.pipeline.W = <root>/weights`).
- `main.py` — thin CLI: `python main.py <audio>` (file → `translations/<stem>_translated.wav` + `.txt`, 3-thread pipelined, ~3× RT) or `python main.py --mic` (encoder/LM/decoder threads; live critical path = LM step ~24 ms on M4).
- `moshi-mlx/` — **vendored fork of `moshi-mlx` 0.3.0** (installed editable). Owns all hibiki-zero deltas: `hidden_scale=6` + `kv_repeat=2` (GQA) + `rope_concat` + the learned per-slice depformer output **LayerNorm** (`depformer_norms.{i}`) — without the LayerNorm the audio babbles (silence-in test: correct model rms≈0.06, buggy rms≈0.13 peak≈1.23). Also handles Hibiki-M's float `hidden_scale=4.125`. **q4 must be regenerated after touching the depformer** (LayerNorm stays bf16; nn.quantize skips non-Linear). Runs on mlx 0.31 despite the fork's `<0.27` pin.
- `scripts/` — conversion/verification/profiling only: `convert_mlx_q4.py` (PyTorch LM → `weights/hibiki.q4.safetensors`, `nn.quantize(bits=4, group_size=32)`), `convert_hibiki_m_mlx_q4.py` (stages `weights/hibiki-m-mlx-q4/`, 1.125 GB q4 LM), `verify_mlx_q4.py` (gate: leon.wav → `translations/leon_mlx_q4.wav`), `profile_mlx.py`, `push_mlx_q4.py`, `push_hibiki_m_mlx_q4.py`. **Keep `group_size=32`** — stock moshi-mlx/moshi-swift hardcode gs32 for `.q4.safetensors`.
- `assets/samples/` — test clips: `leon.wav` (~64 s FR Olympics), `crepes.mp3`, 12 s cuts.
- `weights/` (gitignored) — 3B checkpoint (`config.json`, `hibiki-pytorch-77f82164@110.safetensors` 5.8 GB, `hibiki.q4.safetensors` 2.2 GB, mimi 367 MB, tokenizer) + `hibiki-m-mlx-{bf16,q4}/`.
- `finetune/` — PyTorch LoRA training stack (uses the `moshi` pip package; PyTorch eval helpers live in `finetune/hibiki_helpers.py`, relocated from the deleted upstream `hibiki_zero` package).
- `remote_dataset/` — CoVoST2 + FLEURS eval data & scripts. `training-data/` — PhoMT TTS pipeline. `docs/`, `vision/`.
- Published artifacts: [`huybik/hibiki-zero-3b-mlx-q4`](https://huggingface.co/huybik/hibiki-zero-3b-mlx-q4) (weights + portable stock-pkg patch shim), [`huybik/hibiki-1b-mlx-q4`](https://huggingface.co/huybik/hibiki-1b-mlx-q4) `@9649ed0`.

## Runtime facts (M4 Pro, q4)
- Pipelined file path **~2.7–3.0× RT** (37.5 frames/s): encoder thread streams `encode_step` ahead → main-thread GPU LM step → decoder thread. rustymimi releases the GIL; each thread needs its **own** `rustymimi.Tokenizer` ("Already borrowed" panic otherwise). Codec threads queue **numpy** (not mx) arrays — lazy mx graphs are bound to the creating thread's stream and mlx ≥0.27 refuses cross-thread eval.
- Per-frame GPU LM ≈ 24 ms = main transformer 8.6 ms + **depformer 15.7 ms (64%)**; codec fully hidden. Depformer is **sequential-launch-bound** (16 slices × ~0.72 ms, ~370 launches/frame) — quantization gives zero M4 speedup (q3 quality-safe, q2 breaks); only the parallel-codebook head (refactor plan Phase 3 / distill plan) removes it. `mx.compile` on the main transformer won only ~4% (kernel-bound); pipelined 3× RT is near the practical floor for the 3B on M4.
- Tail flush: input ends → feed `tail_s` (8 s) silence to flush the ~6 s translation lag, early-stop after `PAD_STOP`=12 pad frames (sitting longer hallucinates). Text temp **0.4** default (audio 0.8): kills spurious cold-start openers, BLEU 22.1→25.8 (text feeds back autoregressively; greedy 0.0 over-collapses).
- Gates for runtime changes: `python scripts/verify_mlx_q4.py` coherent, and silence-in (10 s zeros → rms < 0.10, peak < 1.1; healthy model: rms≈0.0002, peak≈0.012).
- Codec-fidelity: decoding generated tokens at fewer codebooks degrades gracefully (mel-dist 0.049@8, 0.105@4). Inference-time depformer truncation is NOT a faithful dep_q=8 preview (feedback goes OOD).

## Benchmarks
- `remote_dataset/download_covost2.py` → wav + manifest; `run_batch.py` (loads LM once, uses `hibiki_mlx`) → text sidecars; `evaluate_translation_text.py` → BLEU/chrF/WER. **q4 CoVoST2 fr→en n=30: BLEU 25.7, chrF 49.4** (paper 30.6; WER is a poor MT metric). If `HF_HOME` points at a dead mount: `export HF_HOME="$(pwd)/.hf_cache"`.
- FLEURS vi↔en (`remote_dataset/fleurs_vi_en/`, downloader joins vi_vn+en_us on FLoRes id): train 1449 / val 149 / test 347 pairs (vi 6.19 h). **vi→en baseline ≈ zero** (BLEU 0.26) — vi is unsupported, needs training.

## Finetune (Track A, Vietnamese)
`finetune/`: `build_pairs.py` → `cache_codes.py` (Mimi/text codes, coarse target delay) → `train_lora.py` (LoRA on `LMModel.transformer`, optional `text_linear` + audio-head LoRA via `--train-text-head`/`--train-audio-heads`, zero-init new LoRA, replay-weighted sampling `--replay-*`, resume) → `validate_lora.py` / `eval_lora.py` (batched greedy eval, loop metrics, BLEU/chrF/WER) → `autoresearch.py` (fixed trial harness, TSV protocol, primary metric val16 chrF).
- Key findings: supervise prefix-delay pads but mask tail padding; text loss weight 5; zero-init audio-head LoRA expansion (random-init spikes loss); best 16-row result 11/16 exact (`vn_lora_short16_audioheads_lr1e4_from_s800` recipe = transformer LoRA + full text_linear + audio-head LoRA, lower-LR continuation). Broad rank32 run underfit; staged replay 300→100 is best fixed-gate row (chrF 9.20, seen3 3/3) but val16 output still loops — not real translation yet.
- Protocol: seen3/short16 for smoke only, 128/512-row eval for real decisions, full 1449 last. MPS training: `--dtype bfloat16 --batch-size 2 --grad-accum-steps 2` (batch 4 spikes to 47.5 GB driver memory); eval can use batch 8. Karpathy autoresearch pattern: one hypothesis per commit, keep only if primary metric improves.
- Next: schedules (text/audio weight, replay, per-group LR), val128 gate, CUDA-clean device handling — refactor plan Phase 4.

## PhoMT synthetic speech (`training-data/`)
Kokoro TTS EN/VI generation (writes outside repo); `af_nicole` at 1.35×, `am_michael` 1.10×; `upload.py` filters EN/VI duration ratio 0.5–1.6.

## Gotchas
- Input must be FR/ES/PT/DE. `sphn.read(sr=24000)` resamples on load, so 16 kHz wavs run directly.
- `nn.quantize` predicate must skip modules whose last dim isn't divisible by 32 (Hibiki-M conditioners).
- HF xethub CDN times out: `HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120`.

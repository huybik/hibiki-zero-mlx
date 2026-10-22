# Environment

Read when setting up a machine, handling credentials, using external storage or pushing.
Back to [CONTEXT.md](../../CONTEXT.md) · [Python](#python) · [Secrets](#secrets) ·
[Storage](#storage) · [Git](#git)

## Python

- Local work uses the conda base interpreter `/opt/homebrew/Caskroom/miniconda/base/bin/python`
  (Python 3.13), which has the inference and data dependencies installed. The macOS app
  hard-codes the same path.
- Inference needs MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and sounddevice for the
  microphone. Install with `pip install -e ./moshi-mlx` then `pip install -e .`.
- CUDA training uses only the environment that `./finetune/h100.sh setup` builds; see
  [sft-training](sft-training.md).

## Secrets

- The gitignored `.env` holds `HF_TOKEN` and `MDC_API_KEY`. Never print, log or commit their
  values. Export them into a command's environment without echoing them.

## Storage

- `/Volumes/data` is the external data drive. `weights/` is a symlink to
  `/Volumes/data/models/hibiki/weights`. That folder holds `config.json`, the Mimi checkpoint,
  the tokenizer and the staged BF16 model `vi-step135000-mlx-bf16/`, but neither
  `hibiki.q4.safetensors` nor the upstream PyTorch LM. `--model 3b` therefore needs the README's
  `hf download` and `scripts/convert_mlx_q4.py` first.
- Corpora live under `/Volumes/data/datasets/`. Only `en_speech/` is built by code on `main`
  ([en-speech-corpus](en-speech-corpus.md)).
- Gitignored generated paths: `translations/`, `finetune/pairs/`, `finetune/cache/`,
  `finetune/runs/` and `remote_dataset/fleurs_vi_en/`.

## Git

- `origin` is `github.com/huybik/hibiki-zero-mlx`; `upstream` is `kyutai-labs/hibiki-zero`.
- Only the `huybik` GitHub account can push to `origin`. If `gh` has another account active,
  run `gh auth switch --user huybik` first.
- Commit messages carry no AI attribution and no `Co-Authored-By` lines (see `AGENTS.md`).

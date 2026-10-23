# Environment

Read when setting up a machine, handling credentials, using external storage or pushing.
Back to [CONTEXT.md](../../CONTEXT.md) · [Python](#python) · [Secrets](#secrets) ·
[Storage](#storage) · [Git](#git)

## Python

- Local work uses the conda base interpreter `/opt/homebrew/Caskroom/miniconda/base/bin/python`
  (Python 3.13), which has the inference and data dependencies installed, including `anthropic`
  and `google-genai` for the translation. The macOS app
  hard-codes the same path.
- Inference needs MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and sounddevice for the
  microphone. Install with `pip install -e ./inference/moshi-mlx` then `pip install -e .`; both
  are editable installs, so moving either directory requires reinstalling them.
- CUDA training uses only the environment that `./training/h100.sh setup` builds; see
  [sft-training](sft-training.md).

## Secrets

- The gitignored `.env` holds `HF_TOKEN`, `MDC_API_KEY`, the paid Gemini API keys
  `GEMINI_API_KEY_1` and `GEMINI_API_KEY_2` (`GEMINI_API_KEY` is the same key as `_2`) and the
  paid OpenRouter key `OPEN_ROUTER`. Never
  print, log or commit their values. Export them into a command's environment without echoing
  them.

## Storage

- `/Volumes/data` is the external data drive. `weights/` is a symlink to
  `/Volumes/data/models/hibiki/weights`. That folder holds `config.json`, the upstream PyTorch
  LM `hibiki-pytorch-77f82164@110.safetensors`, `hibiki.q4.safetensors` (so `--model 3b`
  works), the Mimi checkpoint, the tokenizer and the staged BF16 model
  `vi-step135000-mlx-bf16/`.
- `/Volumes/data/datasets/en_vi_speech/` is the EN-VI dataset holding the English corpus
  ([en-vi-speech-dataset](en-vi-speech-dataset.md)), and `/Volumes/data/datasets/phomt-grounded-v2-5s/`
  the restored PhoMT cache ([sft-data-caches](sft-data-caches.md#phomt-cache)). The FLEURS
  data are deleted.
- Gitignored local state: `data/` (pair files and Mimi caches), `runs/` (training runs and eval
  outputs), `translations/` and `output/`.

## Git

- `origin` is `github.com/huybik/hibiki-zero-mlx`; `upstream` is `kyutai-labs/hibiki-zero`.
- With no default repo set, `gh` prefers the `upstream` remote, so `gh pr` commands fail with
  "Could not resolve to a PullRequest". This Mac's clone has `gh repo set-default
  huybik/hibiki-zero-mlx` in its git config; a fresh clone needs it once, or pass
  `-R huybik/hibiki-zero-mlx`.
- Only the `huybik` GitHub account can push to `origin`. Other sessions may leave another
  account active, so push without switching it: `GH_TOKEN=$(gh auth token --user huybik) git -c
  credential.helper= -c credential.helper='!gh auth git-credential' push …`, and run `gh pr` the
  same way under `GH_TOKEN`.
- Commit messages carry no AI attribution and no `Co-Authored-By` lines.

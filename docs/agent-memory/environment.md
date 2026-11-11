# Environment

Read when setting up a machine, handling credentials, using external storage or pushing.
Back to [CONTEXT.md](../../CONTEXT.md) · [Python](#python) · [Secrets](#secrets) ·
[Storage](#storage) · [Git](#git)

## Python

- Local work uses the conda base interpreter `/opt/homebrew/Caskroom/miniconda/base/bin/python`
  (Python 3.13), which has the inference and data dependencies installed, including `anthropic`
  and `google-genai` for the translation. The macOS app
  hard-codes the same path. Its PyTorch is 2.15.0, the pod's pin, for the MPS eval
  ([sft-evaluation](sft-evaluation.md#side-tools)), with moshi 0.2.13 as on the pod.
- Inference needs MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and sounddevice for the
  microphone. Install with `pip install -e ./inference/moshi-mlx` then `pip install -e .`; both
  are editable installs, so moving either directory requires reinstalling them.
- CUDA training uses only the environment that `./training/h100.sh setup` builds; see
  [sft-recovery](sft-recovery.md).
- Free-threaded Python 3.14 (no GIL) for every box env it can run (user, 2026-11-11: "we use this
  new python setup for everything"), at the `.venv`'s package versions. The cache builds run on it
  (`/workspace/envs/cache`, [cache-build](cache-build.md#speed-on-cuda)). Two limits: importing
  Triton turns the GIL back on, and forcing it off under Triton killed the cache build, so a job on
  `torch.compile` (the trainer) gains nothing from the GIL's absence and must be benchmarked
  against 3.12 first; vLLM and flash-attn ship no free-threaded wheels (the tranche's `nanovox`,
  `asr` and `mt` envs stay on 3.12).
- A rented box shows the host's cores (`nproc` 128) but its container may use only its quota
  (`/sys/fs/cgroup/cpu.max`, 15.36 CPUs on the Illinois H100), and torch (64 threads, 32
  compile workers), ninja and BLAS size themselves by the cores. `setup` ends with
  `h100.sh cpus`, which writes `MAX_JOBS`, `OMP_NUM_THREADS` and `TORCHINDUCTOR_COMPILE_THREADS`
  at the quota into `/etc/environment`, loaded by every ssh session (not a tmux server started
  before it, nor what starts outside a login); rerun it after a container restart. Code that starts several processes splits the quota itself (`training.paths.cpu_budget`).

## Secrets

- The gitignored `.env` holds `HF_TOKEN` (the huybik account), `HF_TOKEN_AN` (a write token of
  the anquachdev account, which owns the dataset `anquachdev/common-vi-en`), `MDC_API_KEY`, the
  paid Gemini API keys
  `GEMINI_API_KEY_1` and `GEMINI_API_KEY_2` (`GEMINI_API_KEY` is the same key as `_2`) and the
  paid OpenRouter key `OPEN_ROUTER`. The H100 box's `/workspace/.env` (mode 600) holds `OPEN_ROUTER`
  and `HF_TOKEN_AN`. Never
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
  the restored PhoMT cache ([sft-data-caches](sft-data-caches.md#phomt-cache)). FLEURS,
  MultiMed-ST and `en-vi-val` are evaluation sets only, built into `data/eval/`
  ([sft-evaluation](sft-evaluation.md#evaluation-sets)).
- Hugging Face downloads (models, the VieNeu and MOSS weights) cache in `/Volumes/data/huggingface`
  (`HF_HOME`).
- The Mac keeps no run checkpoints or evaluation outputs: the pod's validation scores every
  checkpoint on all evaluation sets, as `eval.py` would (user's decision, 2026-11-04). The sets
  themselves are on the Hub ([sft-evaluation](sft-evaluation.md#evaluation-sets)). To score a
  checkpoint on the Mac anyway, put its bf16 copy under
  `/Volumes/data/models/hibiki/runs/<run>/checkpoints/`, never the internal disk (a 3B checkpoint
  is 5.8 GiB in bf16, 11.6 GiB in fp32), and delete it once scored.
- Gitignored local state: `data/` (restored Mimi caches, `data/eval/`), `runs/` (training runs and eval
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

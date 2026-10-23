# Hibiki MLX — project context

Last verified: 2026-10-23

## Purpose

This repo runs Hibiki-Zero simultaneous speech translation:

- MLX inference on Apple Silicon (upstream q4 for FR/ES/PT/DE→EN, or staged BF16 checkpoints);
- full-model Vietnamese→English voice-preserving SFT on one H100 on the 5–22.4 s clips of the
  English read-speech corpus `en_speech` plus the ~0.8k-hour PhoMT cache, together ~10k hours of
  5–15 s clips;
- the roadmap toward VI→EN parity with the upstream languages.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero),
[paper](https://arxiv.org/abs/2602.11072).

## Map

```text
training/  (CUDA, Linux H100; en_speech is built and translated locally by prep/)
  en_speech ─► prep/translate_en_speech ─► (VI TTS pair rows: not built) ─► prep/cache_codes ─► data/cache
    ─► h100.sh train → finetune/train.py ─► runs/, HF recovery repo
  HF PhoMT cache ─► data/cache/phomt-grounded-v2-5s ─────────┘ (one more --cache-dir)
inference/ (Apple Silicon, MLX)
  checkpoints ─► tools/convert_mlx_{q4,bf16}.py ─► weights/… ─► main.py · hibiki_mlx/ · macos/
```

- `inference/` is the only Apple Silicon code: `main.py`, the `hibiki_mlx/` runtime, the
  vendored `moshi-mlx/`, `tools/`, `assets/leon.wav` and the `macos/` test app.
- `training/` holds `h100.sh`, shared `paths.py`, `prep/` (the `en_speech` corpus builder, its
  Vietnamese translation and the pair → grounded cache builder) and `finetune/` (the generic SFT
  trainer, evaluation and HF recovery sync). Its modules run from the repo root as
  `python -m training.<package>.<module>`; everything except the corpus builder and the
  translation runs on CUDA only.
- `docs/` holds plans, analyses and reports.
- Gitignored: `weights/` (shared by both sides), `data/` (pairs, caches), `runs/` (training
  runs and eval outputs) and `translations/` (inference outputs).

## Where things stand

- FLEURS and the PhoMT code and frozen recipe are dropped; the PhoMT grounded cache is kept and
  pooled with the `en_speech` caches. The last run on PhoMT and FLEURS,
  `grounded_v2_full_direct_voice_5epoch`, is analysed up to step 135k in `docs/analysis/`:
  teacher-forced loss improves, but free-running translation fails.
- [docs/vi_en_parity_plan.md](docs/vi_en_parity_plan.md) is the active roadmap. In phase 1,
  the `en_speech` corpus is built and its transcripts are being translated to Vietnamese:
  the first tranche and all Common Voice on Claude
  ([en-vi-translation](docs/agent-memory/en-vi-translation.md)),
  2,000 h of Libriheavy on Gemini ([en-vi-gemini](docs/agent-memory/en-vi-gemini.md)) and the
  rest on Muse ([en-vi-muse](docs/agent-memory/en-vi-muse.md)), but the TTS step that turns it
  into Vietnamese-source pair rows is not, so there are no caches to train on yet.
- GRPO (roadmap phase 4) is planned in [docs/grpo_plan.md](docs/grpo_plan.md) but has no code.

## Essential invariants

- Never print, log or commit `.env` values (`HF_TOKEN`, `MDC_API_KEY`, the Gemini and
  OpenRouter keys) →
  [environment](docs/agent-memory/environment.md).
- Never upload or re-share the Common Voice part of `en_speech` →
  [en-speech-corpus](docs/agent-memory/en-speech-corpus.md).
- Apple Silicon runs inference, the `en_speech` corpus builder and its translation only; Mimi
  caching, training and evaluation run on CUDA → [sft-training](docs/agent-memory/sft-training.md).
- Production runs go only through `training/h100.sh train`, which ties a run to its commit and
  HF recovery prefix → [sft-training](docs/agent-memory/sft-training.md).
- Cache rows are `[33, T]`: text, 16 English target codebooks, then 16 Vietnamese source
  codebooks. EOS and -1 positions are fixed, and training applies no transform →
  [sft-data-caches](docs/agent-memory/sft-data-caches.md).
- Resume only from the newest complete model and optimizer pair of the same run →
  [sft-training](docs/agent-memory/sft-training.md).
- In MLX, give each thread its own `rustymimi.Tokenizer` and pass NumPy arrays, not MLX arrays,
  between threads → [mlx-inference](docs/agent-memory/mlx-inference.md).
- q4 means 4-bit, group size 32, on layers whose input width divides by 32. A BF16 model
  directory needs `weight_dtype: bfloat16` → [mlx-inference](docs/agent-memory/mlx-inference.md).

## Topic Index

| Read when working on… | Note | Main paths |
|---|---|---|
| MLX inference, conversion, benchmarks, macOS app | [mlx-inference](docs/agent-memory/mlx-inference.md) | `inference/` |
| SFT trainer, H100 launcher, checkpoints, recovery, eval | [sft-training](docs/agent-memory/sft-training.md) | `training/h100.sh`, `training/finetune/` |
| Pair rows, grounded Mimi caches, CTC timing | [sft-data-caches](docs/agent-memory/sft-data-caches.md) | `training/prep/cache_codes.py`, `training/prep/text_timing.py`, `training/paths.py` |
| English read-speech corpus | [en-speech-corpus](docs/agent-memory/en-speech-corpus.md) | `training/prep/build_en_speech.py`, `/Volumes/data/datasets/en_speech` |
| Roadmap, GRPO plan, analyses, reports | [docs-and-plans](docs/agent-memory/docs-and-plans.md) | `docs/` |
| Python, secrets, external storage and corpora, pushing | [environment](docs/agent-memory/environment.md) | `.env`, `weights/`, `data/`, `runs/`, `/Volumes/data` |
| EN→VI transcript translation | [en-vi-translation](docs/agent-memory/en-vi-translation.md) | `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text` |
| EN→VI translation on the Gemini Batch API, claims | [en-vi-gemini](docs/agent-memory/en-vi-gemini.md) | the `claim` and `gemini` steps of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/gemini` |
| EN→VI translation on Muse through OpenRouter | [en-vi-muse](docs/agent-memory/en-vi-muse.md) | the `muse` step of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/muse` |

## Canonical resources

- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
- Run recovery repo (`HIBIKI_HF_REPO`): https://huggingface.co/huybik/hibiki-zero-vi-full-sft

## Maintaining context

```text
CLAUDE.md (imports this file)
  └─ CONTEXT.md: orientation, invariants, Topic Index
       └─ docs/agent-memory/<topic>.md: one note per aspect (what is true and how to work on it),
          read when work touches its paths
```

- Record current facts checked against the code, and guidance on how to work (decisions,
  feedback) with its reason. This replaces each agent's local memory: save here what you would
  have saved there, so everyone shares it. Link to existing docs instead of copying, keep one
  owner per fact, and keep no history.
- Update the owning note in the same commit as the change. Knowledge with no owner gets a new
  note, added to the Topic Index.
- Change this file only when orientation, an invariant or the index changes, and update
  `Last verified` when you do.
- Keep this file under ~1,500 words and each note under ~1,000. Split a subject into a new note
  before one grows past that.
- Each note opens with `Read when touching …`, a link back here, and links to its sections.

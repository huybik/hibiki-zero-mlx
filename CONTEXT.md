# Hibiki MLX — project context

Last verified: 2026-10-22

## Purpose

This repo runs Hibiki-Zero simultaneous speech translation:

- MLX inference on Apple Silicon (upstream q4 for FR/ES/PT/DE→EN, or staged BF16 checkpoints);
- direct full-model Vietnamese→English voice-preserving SFT on one H100;
- a 12-layer mobile student, and the roadmap toward VI→EN parity with the upstream languages.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero),
[paper](https://arxiv.org/abs/2602.11072).

## Map

```text
FLEURS, PhoMT ─► finetune/ caches ─► finetune/h100.sh → train.py (H100 SFT) ─► HF recovery repo
                                              │
                                              └► finetune/grpo_rollout.py (GRPO, phase A only)
3B teacher ─► student/ (12 layers + parallel_v1 head) ─► BF16 export
checkpoints ─► scripts/convert_mlx_{q4,bf16}.py ─► main.py · hibiki_mlx/ · moshi-mlx/ · macos/
remote_dataset/build_en_speech.py ─► /Volumes/data/datasets/en_speech (roadmap phase 1 input)
```

- `main.py`, `hibiki_mlx/`, `moshi-mlx/`, `scripts/` and `macos/HibikiTestApp/` are inference.
- `finetune/` holds CUDA SFT, caches, evaluation and GRPO.
- `student/` is mobile-student distillation.
- `remote_dataset/` holds dataset downloaders and builders.
- `docs/` holds plans, analyses and reports.

## Where things stand

- The 5-epoch SFT run `grounded_v2_full_direct_voice_5epoch` is analysed up to step 135k.
  Teacher-forced loss improves, but free-running translation fails, and the FLEURS validation
  loss diverges.
- [docs/vi_en_parity_plan.md](docs/vi_en_parity_plan.md) is the active roadmap. In phase 1,
  the English source corpus is built, but the translation, TTS and caching pipeline is not.
- GRPO phase A is coded but has a reward defect. The student code is complete but has not been
  run.

## Essential invariants

- Never print, log or commit `.env` values (`HF_TOKEN`, `MDC_API_KEY`) →
  [environment](docs/agent-memory/environment.md).
- Never upload or re-share the Common Voice part of `en_speech` →
  [en-speech-corpus](docs/agent-memory/en-speech-corpus.md).
- The SFT recipe constants live in `freeze_full_data_receipt.py`, and `train.py` rejects any
  deviation. Production runs go only through `h100.sh` → [sft-training](docs/agent-memory/sft-training.md).
- Cache rows are `[33, T]`: text, 16 English target codebooks, then 16 Vietnamese source
  codebooks. EOS and -1 positions are fixed, and training applies no transform →
  [sft-data-caches](docs/agent-memory/sft-data-caches.md).
- Resume only from the newest complete model and optimizer pair of the same run →
  [sft-training](docs/agent-memory/sft-training.md), [mobile-student](docs/agent-memory/mobile-student.md).
- In MLX, give each thread its own `rustymimi.Tokenizer` and pass NumPy arrays, not MLX arrays,
  between threads → [mlx-inference](docs/agent-memory/mlx-inference.md).
- q4 means 4-bit, group size 32, on layers whose input width divides by 32. A BF16 model
  directory needs `weight_dtype: bfloat16` → [mlx-inference](docs/agent-memory/mlx-inference.md).
- The student computes in BF16 but keeps FP32 masters, optimizer state and recovery
  checkpoints, casting to BF16 only at export. Every stage is bound to the previous one by
  SHA-256 → [mobile-student](docs/agent-memory/mobile-student.md).
- Stock moshi-swift cannot load the `parallel_v1` head →
  [mlx-inference](docs/agent-memory/mlx-inference.md).
- GRPO rewards are currently identical within a group, so every advantage is 0; fix this before
  phase B → [grpo](docs/agent-memory/grpo.md).

## Topic Index

| Read when working on… | Note | Main paths |
|---|---|---|
| MLX inference, conversion, benchmarks, macOS app | [mlx-inference](docs/agent-memory/mlx-inference.md) | `main.py`, `hibiki_mlx/`, `moshi-mlx/`, `scripts/`, `macos/` |
| SFT recipe, H100 launcher, checkpoints, recovery, eval | [sft-training](docs/agent-memory/sft-training.md) | `finetune/{train,common,hf_sync,eval,validate}.py`, `finetune/h100.sh` |
| Pairs, Mimi caches, publishing, frozen receipt | [sft-data-caches](docs/agent-memory/sft-data-caches.md) | `finetune/{cache_codes,cache_phomt_stream,text_timing,build_pairs,publish_grounded_cache,freeze_full_data_receipt}.py`, `remote_dataset/download_*.py` |
| GRPO rollouts, rewards, trainer | [grpo](docs/agent-memory/grpo.md) | `finetune/grpo_rollout.py`, `finetune/tests/`, `docs/grpo_plan.md` |
| Mobile student distillation and export | [mobile-student](docs/agent-memory/mobile-student.md) | `student/`, `docs/mobile_student_plan.md` |
| English read-speech corpus | [en-speech-corpus](docs/agent-memory/en-speech-corpus.md) | `remote_dataset/build_en_speech.py`, `/Volumes/data/datasets/en_speech` |
| Roadmap, analyses, reports, proposals | [docs-and-plans](docs/agent-memory/docs-and-plans.md) | `docs/` |
| Python, secrets, external storage, pushing | [environment](docs/agent-memory/environment.md) | `.env`, `weights/`, `/Volumes/data` |

## Canonical resources

- Caches: https://huggingface.co/datasets/huybik/hibiki-zero-vi-full-sft/tree/main/grounded-v2
- Recovery: https://huggingface.co/huybik/hibiki-zero-vi-full-sft/tree/main/grounded_v2_full_direct_voice_5epoch
- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
- PhoMT speech data: https://huggingface.co/datasets/anquachdev/PhoMT-en-vi-speech

## Maintaining context

```text
CLAUDE.md (imports AGENTS.md + this file)
  └─ CONTEXT.md: orientation, invariants, Topic Index
       └─ docs/agent-memory/<topic>.md: one note per aspect, read when work touches its paths
```

- Record current facts checked against the code. Link to existing docs instead of copying,
  keep one owner per fact, and keep no history.
- Update the owning note in the same commit as the change. Knowledge with no owner gets a new
  note, added to the Topic Index.
- Change this file only when orientation, an invariant or the index changes, and update
  `Last verified` when you do.
- Keep this file under ~1,500 words and each note under ~1,000. Split a subject into a new note
  before one grows past that.
- Each note opens with `Read when touching …`, a link back here, and links to its sections.

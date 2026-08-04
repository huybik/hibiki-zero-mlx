# Retry-v6 unattended cache and publication queue

Date: 2026-08-04. This records an authorized queue, not a completed cache or publication result.

## Authorization and boundary

The user explicitly requested that the necessary work be queued so the complete dataset can be uploaded unattended. `finetune/complete_vivos_release.py` implements that request without weakening machine gates. Its launch-time SHA-256 is `537483fe85755f8125e03cd2728e66428b1736994efc59a7a76aec0ac104b798`.

The worker waits for the existing guarded supervisor. It proceeds only from `pending_manual_review` when the terminal selection is machine `go` and every frozen machine check passes. It then writes an immutable, scope-bound waiver that explicitly waives only manual listening. Machine `no_go`, a halted supervisor, changed scripts/contracts, invalid provenance, cache audit failure, packaging failure, a non-empty remote prefix, upload mismatch, or clean-room extraction failure halts the queue before later stages.

The ordered stages are:

1. Wait for attempt-0 completion, exact validation, pinned MPS QA, and frozen retry selection.
2. Bind the explicit unattended manual-listening waiver to the repaired production plan, terminal selection, and required-candidate digest.
3. Rerun finalization and require final status `go`.
4. Run final provenance preflight.
5. Build resumable PyTorch-Mimi v2 shards on MPS and require the full cache audit.
6. Independently preflight and package the immutable release.
7. Upload the fixed Hub prefix and require remote file/LFS hashes plus clean-room download, extraction, and cache audit.

The waiver does not relabel failed QA, skip rejected rows, or claim that listening occurred. It is packaged as the release's manual evidence.

## Preflight

| Check | Result |
|---|---|
| Runtime | Python 3.13 base; torch 2.13.0; huggingface-hub 1.21.0; zstandard 0.23.0 |
| Token handling | `.env` mode 0600; exactly one `HF_TOKEN`; value not logged or hashed |
| Dataset repository | `huybik/hibiki-zero-vi-full-sft` |
| Remote prefix | `v2/vivos_qwen3_tts_mlx_retry_v6_full/`; zero existing entries at HEAD `0dc2ef1c2be9473393fc9d4b705d9b98c83c7b43` |
| Disk | 245 GiB free on `/Volumes/data` |
| Cache destination | absent |
| Release destination | absent |
| Mimi/config/tokenizer weights | present at the repository defaults |
| Generator | repaired attempt 0 live |
| Postprocess supervisor | `waiting_attempt0` |

## Fixed paths

- Plan: `/Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan_repair1.json`
- QA: `/Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/`
- Queue records: QA root `unattended_release_repair1/`
- Cache: `/Volumes/data/datasets/hibiki_vi_v2/cache/vivos_qwen3_tts_mlx_retry_v6_mimi_v2/`
- Release: `/Volumes/data/datasets/hibiki_vi_v2/releases/v2/vivos_qwen3_tts_mlx_retry_v6_full/`

The queue config binds its repository commit, repaired plan, supervisor config, scripts, gender files, destinations, and authorization. Events and command results are append-only; every stage has a timestamped combined log. Successful publication is represented only by the release's immutable `release_report.json`, including the Hub commit OID and clean-room result.

## Launch command

```bash
tmux new-session -d -s hibiki_vivos_qwen_v6_release_repair1_20260804 \
  "zsh -lc 'cd /Users/macoblle/MEGA/Projects/sidequest/research/hibiki-zero/code && exec /opt/homebrew/Caskroom/miniconda/base/bin/python finetune/complete_vivos_release.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan_repair1.json --qa-root /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full --supervisor-work /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/supervisor_repair1 --work-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/unattended_release_repair1 --cache-root /Volumes/data/datasets/hibiki_vi_v2/cache/vivos_qwen3_tts_mlx_retry_v6_mimi_v2 --dataset-root /Volumes/data/datasets/hibiki_vi_v2 --gender-files /Volumes/data/datasets/hibiki_vi_v2/raw/vivos/corpus/vivos/train/genders.txt /Volumes/data/datasets/hibiki_vi_v2/raw/vivos/corpus/vivos/test/genders.txt --env-file /Users/macoblle/MEGA/Projects/sidequest/research/hibiki-zero/code/.env --poll-seconds 30 > /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/unattended_release_repair1/worker.log 2>&1'"
```

At queue preparation time no waiver, Mimi cache, release directory, Hub commit, or upload had been created.

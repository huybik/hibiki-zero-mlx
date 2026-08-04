# Reproduction commands

Pinned runtime: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python` for MLX model execution; conda base for preparation and static verification.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_batch.py prepare-benchmark /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/generation_plan.jsonl --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v1_benchmark_2026-08-04

/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_batch.py benchmark /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v1_benchmark_2026-08-04/cohort_plan.jsonl --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v1 --device mps

/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_batch.py prepare-production /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/generation_plan.jsonl --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v1_full --batch-size 8
```

The production command was deliberately not run. The prepared plan contains 10,950 rows in 1,391 same-speaker, target-length-sorted batches.

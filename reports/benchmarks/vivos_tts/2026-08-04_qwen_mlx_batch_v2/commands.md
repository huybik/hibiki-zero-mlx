# Reproduction commands

Pinned MLX runtime: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python` (`mlx-audio==0.4.7`, commit `2c9461f5d8315fa8e7013ab2729495b2bb83d384`, MLX 0.32.0). Pinned QA runtime: `/Volumes/data/envs/hibiki-vivos-qa/bin/python` (transformers 4.57.3, Whisper `41f01f3…`, WavLM `feb593a…`). Ordinary checks use conda base.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_batch_v2.py prepare-benchmark /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/generation_plan.jsonl --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04

/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_batch_v2.py benchmark /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v2 --device mps

/Volumes/data/envs/hibiki-vivos-qa/bin/python training-data/qa_vivos_qwen_mlx_batch_v2.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --benchmark-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04 --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_batch_v2 --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v2 --device mps

/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/profile_vivos_qwen_mlx_optimizations.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --benchmark-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04 --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2_optimization --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v2

/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_quantized.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2_q4 --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v2

/Volumes/data/envs/hibiki-vivos-qa/bin/python training-data/qa_vivos_qwen_mlx_batch_v2.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --benchmark-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2_q4 --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_batch_v2_q4 --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_batch_v2/q4_quality --device mps --skip-batch16
```

The first v2 preflight plan was preserved externally as `vivos_qwen3_tts_mlx_batch_v2_superseded_preflight_2026-08-04`: validation exposed that production resume needed schema-aware validation before any audio was generated. The corrected immutable cohort is `v2r1`. B32 and B64 were deliberately not run after complete B16 throughput fell 28.7% below B8, satisfying the frozen clear-saturation stop rule. No production campaign was launched.

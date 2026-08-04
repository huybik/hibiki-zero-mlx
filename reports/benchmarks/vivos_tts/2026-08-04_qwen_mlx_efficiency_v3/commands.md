# Reproduction commands

Pinned model runtime: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python`. Pinned QA runtime: `/Volumes/data/envs/hibiki-vivos-qa/bin/python`. Static/report work: `/opt/homebrew/Caskroom/miniconda/base/bin/python`.

```bash
# Fresh bf16 and exact cached-prefix runs (use --rows 16 or 64).
/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_efficiency_v3.py run-candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3 --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_efficiency_v3 --candidate bf16_prefix --rows 64

# Replace CANDIDATE with main_mlp_q8_g64, main_all_q8_g64,
# code_predictor_q8_g64, or combined_q8_g64.
/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_efficiency_v3.py run-candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3 --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_efficiency_v3 --candidate CANDIDATE --rows 16

/Volumes/data/envs/hibiki-vivos-qa/bin/python training-data/qa_vivos_qwen_mlx_efficiency_v3.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3/CANDIDATE_n16 --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_efficiency_v3/CANDIDATE_n16

/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_efficiency_v3.py compare-exact --baseline /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3/bf16_n64 --candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3/bf16_prefix_n64 --out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_efficiency_v3/prefix_exactness.json

/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/benchmark_vivos_qwen_mlx_decoder_v3.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl --baseline-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3/bf16_prefix_n64 --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_efficiency_v3/decoder --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_efficiency_v3 --groups 4 --static-only
```

The CPU pipeline used the final command without `--static-only`. It was deliberately terminated after 2/32 outputs once elapsed wall time exceeded 264 seconds. No full quantized run, q6/q4 run, second-Metal-stream run, Core ML conversion, production campaign, or upload was launched.

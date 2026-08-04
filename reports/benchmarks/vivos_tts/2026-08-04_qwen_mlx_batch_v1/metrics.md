# Qwen3-TTS MLX same-speaker batch benchmark

Date: 2026-08-04 · Machine: Apple Silicon `arm64` with 48 GiB unified memory · Cohort: 64 frozen rows / 4 speakers.

| Batch | Rows | Wall s | Rows/min | Audio s / wall s | Peak GB | Acoustic failures |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 410.132 | 9.363 | 0.517 | 8.909 | 3 |
| 2 | 64 | 304.585 | 12.607 | 0.678 | 9.021 | 6 |
| 4 | 64 | 244.735 | 15.690 | 0.863 | 9.044 | 3 |
| 8 | 64 | 162.577 | 23.620 | 1.281 | 9.219 | 3 |

Provisional throughput winner: **None**. This is not a corpus-generation quality approval: all generated WAVs, hashes, references, texts, source durations, and acoustic checks are preserved for the existing English-ASR WER and speaker-similarity QA. The scalar figures in `benchmark_summary.json` reuse immutable stopped-campaign sidecar timings and are not a fresh controlled wall-time baseline.

The timed path uses installed `Model.batch_generate`, one frozen reference per same-speaker batch, target-character sorting, fixed group order, and one SHA-256-derived RNG seed per batch. Reference prompts were primed before timing; first-shape compilation remains included. Raw batch records, peak MLX memory, output hashes, environment, commands, and failures are archived beside this report.

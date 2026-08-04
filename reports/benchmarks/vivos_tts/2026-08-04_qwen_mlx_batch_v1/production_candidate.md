# Production candidate decision

Batch 8 is the throughput leader: 23.620 rows/min and 1.281 generated-audio seconds per wall second, with 9.219 GB peak MLX memory. That is 2.52× the controlled batch-1 rows/min and 2.13× the historical scalar sidecar rows/min.

It is **not a quality go**. Three of 64 batch-8 rows missed the coarse source/target duration-ratio band; the benchmark did not run English ASR WER or speaker-embedding similarity. Those rows are retry candidates under the existing QA policy, so the strict benchmark summary correctly remains `no_go` and is not relabeled.

A separate immutable batch-8 production plan and atomic/resumable runner were prepared under `/Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v1_full`. Full generation was not launched. Before replacing the scalar campaign, score the preserved batch-8 WAVs with the existing target QA and approve or reject the new revision on WER, prompt leakage, speaker similarity, and listening—not throughput alone.

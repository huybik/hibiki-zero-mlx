# Qwen3-TTS MLX end-to-end optimization v2

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 revision `a6eb4f68…` · fresh 64-row same-speaker throughput cohort (`VIVOSSPK04`) plus the immutable v1 four-speaker quality cohort.

| Candidate | Rows/min | Gain vs bf16 B8 | Peak MLX GB | Talker/prefill s | Decode s | Automatic QA |
|---|---:|---:|---:|---:|---:|---|
| bf16 B8 | 38.619 | baseline | 8.704 | 69.02 | 30.41 | no-go: WER 0.2343, cosine 0.8973 |
| bf16 B16 | 27.544 | −28.7% | 9.093 | 84.92 | 54.49 | no-go: WER 0.2707, cosine 0.8961 |
| B8 + speaker cache | 36.008 | −6.8% | 8.704 | 73.49 | 33.16 | exact 64/64 vs bf16; inherits no-go |
| B8 + retained allocator cache | 33.903 | −12.2% | 12.984 | 76.28 | 36.98 | exact 64/64 vs bf16; inherits no-go |
| B8 + full reference/prefix cache | 40.388 | +4.6% | 8.886 | 70.10 | 24.98 | exact 64/64 vs bf16; inherits no-go |
| B8 + q4 talker | 59.279 | +53.5% | 6.093 | 37.23 | 27.55 | no-go: WER 0.3313, cosine 0.8739 |

The quality decision is **no-go**, so no full production plan was prepared. On the original four-speaker quality cohort, bf16 B8 also failed automatic quality (WER 0.2439 despite cosine 0.9269 and zero prompt leakage). Manual listening remains required, but it cannot override the current automatic no-go.

The bottleneck ranking is: recurrent talker/code-predictor work first (69% of bf16 B8 wall time), sequential vocoder decode second (31%), then reference preparation. Direct `mx.compile` at the recurrent module boundaries failed because mutable `KVCache` is not a valid compiled argument; internal RoPE/SwiGLU and the vocoder are already compiled. Batched padded vocoder decode was 0.36× the serial path and differed by at most `2.46e-6`, so it is rejected. Attention-mask preallocation makes that isolated micro-operation 5.5× faster, but mask growth plus current sampling accounts for under 0.4% of observed talker time and does not justify a generation-loop rewrite.

All runs stayed far below the 36 GiB active-working-set limit. macOS reported no thermal or performance warning throughout. Existing system swap was already 15–20 GiB and grew during repeated model loads, so the report does not claim a swap-free host. Full prefix caching was the only exact-output positive bf16 lever (median per-group gain 8.2%, aggregate 4.6%); q4 is the large speed/memory win but is rejected as a new synthesis model on quality.

The three v1 duration misses were not token-cap truncation: no row reached its per-sequence cap, and the same rows passed under scalar/B1 or B4. They are stochastic/natural-rate outliers, including a three-word English target paired with 4.625 seconds of Vietnamese source; no shared batch-boundary fix was justified.

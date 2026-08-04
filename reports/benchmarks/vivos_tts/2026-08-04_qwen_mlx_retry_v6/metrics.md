# Qwen3-TTS MLX retry validation v6

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · untouched 64-row, 8-speaker validation cohort.

| Stage | Rows | Generation s | Decode s | Total s | Rows/min | Talker/predictor lane steps | Peak MLX GiB | Thermal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| attempt0_t08 | 64 | 136.60 | 12.03 | 149.39 | 25.70 | 4690/69390 | 6.41 | no warning |
| retry1_t07 | 21 | 80.98 | 4.13 | 85.45 | 14.75 | 1282/18930 | 5.93 | no warning |

| QA pass | Rows | Wall s |
|---|---:|---:|
| attempt0_t08 | 64 | 197.72 |
| retry1_t07 | 21 | 62.94 |

Final decision: **GO**. Accepted 50/64; selected WER 0.07990 (66/826), median speaker cosine 0.91954, prompt leaks 0.

Attempt-0 accepted throughput: 18.48 rows/min generation-only and 7.95 including QA. Final accepted throughput: 12.77 rows/min including retry generation and 6.05 including retry QA (9.91 s/accepted row).

Measured local-compute cost was 234.84 s of candidate generation/decode plus 260.66 s of QA, 495.49 s total. Incremental cloud/API cost was **$0**; electrical energy was not instrumented and no energy claim is made.

The preregistered stop fired after retry round 1, so retry round 2 was not run. A production plan covering 10,950 rows in 1,391 atomic/resumable groups was then created but production generation was deliberately not launched. Its companion attestation binds the exact policy/seeds, group digest, package and `mlx-audio` revisions, 14 model-file hashes, engine/helper/runner hashes, and attempt-manifest contract; the production entrypoint validated every binding before launch.

The policy, cohort, attempts, ASR transcripts, selection, exact commands, hashes, timing, lane accounting, memory, thermal snapshots, and failures are preserved here; WAV/code attempts remain on the external dataset disk.

# Qwen3-TTS MLX deterministic compaction v5

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 `a6eb4f68…` · frozen B8 same-speaker throughput and 8-speaker quality cohorts.

| Candidate | Rows/min | Gain | Generation s | Decode s | Audio s/wall s | Talker/predictor waste | Peak MLX GiB | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| installed B8, group-global RNG | 40.512 | baseline | 66.33 | 28.46 | 2.13 | 18.9%/20.9% | 8.28 | control |
| row-owned RNG, original groups | 42.930 | +6.0% | 78.94 | 10.11 | 2.36 | 24.6%/26.4% | 5.92 | deterministic control |
| fitted-length groups, no compaction | 43.499 | +1.3% | 79.07 | 8.82 | 2.39 | 26.6%/28.3% | 5.92 | reject: waste worsened |
| active-lane compaction | 44.909 | +3.2% | 76.88 | 8.23 | 2.46 | 0.0%/0.0% | 5.94 | advance |
| compaction + compiled predictor | 47.328 | +5.4% | 72.88 | 7.89 | 2.60 | 0.0%/0.0% | 5.94 | additive gain +5.4% |

On the eight-speaker n=64 quality cohort, compact+compiled improved total throughput **24.774→28.407 rows/min (+14.7%)**. Attempt-0 quality was paired non-inferior: WER 0.1092→0.1092; median speaker cosine 0.9310→0.9310; zero prompt leaks for both. The compiled predictor is exact against eager compaction on 64/64 code arrays and WAV hashes.

Quality-adjusted selection retried 13 failing rows with distinct row-owned attempt-1 keys, preserved every attempt, and selected the lowest WER among candidates passing waveform/duration/leak/speaker gates. It accepted 58/64, rejected 6, and selected attempt 1 for 6 rows. Selected WER is **0.1029**, above the frozen 0.08 gate, so the decision is **NO-GO** and no production plan/campaign was created.

Accepted throughput is 20.903 rows/min for the bf16 B8 control, 23.525 rows/min for optimized attempt 0 (+12.5%), and 18.113 rows/min after retry cost. Fresh QA cost 171.499 s separately; including it gives 9.570 selected rows/min.

The frozen length model used all 1,489 immutable scalar sidecars and held out 307 rows (MAE 6.65 codec frames / about 0.53 s), but its schedule increased dead-lane waste from 24.6% to 26.6%; it is rejected as a scheduling improvement. Temperature 0.7 reduced n=16 row failures but regressed n=64 WER (0.1092→0.1126), median cosine, and total throughput; it is rejected. Compaction changes Metal batch width after EOS: 58/64 outputs were bit-exact versus no compaction, with six legitimate stochastic trajectories diverging under batch-width numeric drift. Pinned QA, not bit identity, therefore owns the quality decision.

Every timed candidate used one complete excluded B8 generation+decode warm-up. `pmset -g therm` reported no thermal, performance, or CPU-power warning in every group; raw group system snapshots, active widths/lane-steps, RSS, warm-up times, and model/source hashes are retained in the external candidate records and `raw_timing.jsonl`.

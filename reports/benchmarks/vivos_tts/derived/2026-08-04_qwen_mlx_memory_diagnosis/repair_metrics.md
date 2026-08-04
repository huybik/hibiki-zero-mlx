# Qwen3-TTS MLX allocator-cache repair

Follow-up executed 20:37–20:51 ICT on 2026-08-04. The original diagnosis remains unchanged because its hash is embedded in the repair contract.

## Result

The old supervisor was stopped first, then the generator was interrupted. Its atomic stop boundary contained **668 groups / 5,266 rows**, no hidden temporary directory, and zero media-error rows under the full CPU validator. The original plan, attestation, groups, WAVs, codes, logs, and stopped supervisor directory remain unchanged.

Commit `215467e` places `mx.clear_cache()` in a `finally` block around each production group and corrects new-row `token_count` to `codes.shape[1]`. New group records declare `allocator-cache-repair1`; validators preserve the first 668 groups under their explicit legacy batch-axis semantics. A separate immutable plan and attestation bind the repair:

| Artifact | SHA-256 |
|---|---|
| `production_plan_repair1.json` | `10e3957e750cc4f44781619418cdfc6e601522e130461dde5214ec149be911dc` |
| `production_attestation_repair1.json` | `cc5b1694a02c8a4b4b5bbb50c5c77de0bfe1021b765db17db353ac761f056e10` |

Both the CPU completion validator and pinned MLX runner validator accepted the repaired 10,950-row / 1,391-group contract before restart. Four early low-water samples after 13 new groups measured **7.75, 7.79, 7.76, and 8.27 GiB physical footprint**, with **7.33, 7.38, 7.34, and 7.85 GiB graphics**, versus the old sustained ~35/~34 GiB.

A 20-second time series after 23 new groups separated active working memory from retained memory. Footprint rose from 7.79 GiB to 15 GiB during generation, then fell from **15 GiB at 20:54:45 to 6.64 GiB at 20:54:46** when the group boundary cleared the allocator; the next nine samples held at 7.65 GiB before the following group grew. The process peak remained 16 GiB. This direct rise-and-release trace confirms that working allocations are released per group and the old 35 GiB retained plateau no longer accumulates.

The mixed prefix was validated live at 679 committed groups / 5,349 rows with zero media errors. One hidden temporary group was correctly reported because validation overlapped atomic generation of group 680. Across the first 11 repaired groups (83 rows), corrected token counts ranged from 9 to 128 codec frames. Their summed group time implies 30.03 rows/min, but that short, duration-dependent slice is **not** a throughput benchmark and supports no speed-regression claim.

Generation now runs in `hibiki_vivos_qwen_v6_attempt0_repair1_20260804`. The new supervisor `hibiki_vivos_qwen_v6_postprocess_repair1_20260804` is bound to the repaired plan and worker PID and is in `waiting_attempt0`; it will still stop before cache/publication at the manual-review gate.

One failed read-only inspection is retained: the first post-repair timing-summary script treated `production_*` group IDs from the plan as attempt-directory names and raised `FileNotFoundError`. Replacing only that prefix with `attempt0_t08_` produced the 11-group summary. Generation was unaffected. Exact structured evidence is in `repair.json`, appended `raw_samples.jsonl`, `bindings.json`, and the archived one-shot contract constructor.

## Live speed observation

At 21:19:46 ICT, generation had completed 754/1,391 groups and 5,936 rows. The latest 25 groups ran at **20.03 rows/min and 2.72× realtime**; the latest 50 ran at **19.92 rows/min and 2.68× realtime**.

The raw row rate is not comparable to the pre-fix headline without duration normalization: the latest 50 repaired groups average 8.07 seconds of generated audio per row, while the final 50 pre-fix groups average only 3.63 seconds. Their raw rates are therefore 19.92 versus 39.36 rows/min, but audio throughput is **2.68× versus 2.38× realtime**, or 12.6% higher after repair. This is an observational, speaker/workload-dependent comparison—not a controlled speed A/B—but it provides no evidence that per-group cache clearing slowed duration-normalized synthesis.

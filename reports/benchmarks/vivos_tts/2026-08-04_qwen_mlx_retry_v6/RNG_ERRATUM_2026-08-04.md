# Retry-v6 RNG provenance erratum

Date: 2026-08-04

This is a prose-only provenance correction. It does not modify or relabel the frozen retry-v6 policy, validation result, generated candidates, or selection.

The executable row-root formula in `qwen_mlx_compaction.row_root_digest` is:

`SHA256(f"hibiki-qwen-mlx-row-rng-v1\0{campaign_revision}\0{row_id}\0attempt={attempt}")`

The frozen policy abbreviated the final field as `NUL + attempt`; the executable payload includes the literal `attempt=` prefix. Its properties mention attempt 1, but the same formula also makes attempt 2 distinct. The exact helper SHA-256 is `cb96149414e1c991c0ea29908b3d99a02dd73a12dcd849fde3d6e025eb5dbe82`.

Production finalization records both the untouched frozen prose and the executable formula, helper hash, and derived row-root digest for every selected candidate.

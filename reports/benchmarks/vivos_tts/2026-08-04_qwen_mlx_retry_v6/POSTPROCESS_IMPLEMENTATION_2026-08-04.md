# Retry-v6 postprocess implementation record

Date: 2026-08-04. No MLX or PyTorch model was loaded and no running attempt-0 artifact was modified during this phase.

## Commands

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m py_compile training-data/validate_vivos_qwen_production_v6.py training-data/qa_vivos_qwen_production_v6.py training-data/run_vivos_qwen_postprocess_v6.py
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/validate_vivos_qwen_production_v6.py production /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan.json --attempt 0
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/validate_vivos_qwen_production_v6.py historical-validation /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6/policy.json /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6/attempt0_t08 --attempt 0 --qa-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6/attempt0_t08 --selection-report reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_retry_v6/selection_round1.json
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/validate_vivos_qwen_production_v6.py historical-validation /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6/policy.json /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6/retry1_t07 --attempt 1 --retry-manifest reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_retry_v6/retry_round1.jsonl --qa-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6/retry1_t07 --selection-report reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_retry_v6/selection_round1.json
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/qa_vivos_qwen_production_v6.py audit-historical-selection /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6/policy.json --qa-root /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6 --selection-report reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_retry_v6/selection_round1.json
/Volumes/data/envs/hibiki-vivos-qa/bin/python training-data/qa_vivos_qwen_production_v6.py score-attempt /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan.json --attempt 0 --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/attempt0_t08 --device mps
/opt/homebrew/Caskroom/miniconda/base/bin/ruff check training-data/validate_vivos_qwen_production_v6.py training-data/qa_vivos_qwen_production_v6.py training-data/run_vivos_qwen_postprocess_v6.py
```

## Evidence

- The live production validator returned `incomplete`, not `invalid`, while a single atomic temporary group was present. Successive audits verified 645/82, 735/94, 1,046/133, and 1,142/145 completed rows/groups with zero media errors; progress continued after each snapshot.
- The completed validation attempt 0 passed an exact 64-row / 8-group candidate, WAV, code, QA, and selection-attestation audit.
- The completed retry attempt 1 passed an exact 21-row / 7-group audit.
- Independent recomputation of the frozen selection reproduced 50/64 accepted, 66/826 word errors, WER 0.07990314769975787, median cosine 0.9195427894592285, zero prompt leaks, and `GO`.
- The full scorer invocation stopped before model loading because production generation was incomplete. This expected failure proves the generation-completion gate.
- A temporary-copy corruption probe replaced one copied NPY with invalid bytes. The validator returned explicit `codes_hash` and `media_read:ValueError` reasons; the original candidate files were untouched.

## Failed implementation assumptions retained

- The first CPU validator draft assumed code arrays were `[T,16]`. Inspection showed the generator stores `[1,T,16]` and its existing `token_count` records the leading dimension. The validator was corrected to require a finite, nonempty `[1,T,16]` array without changing any production artifact.
- The conda base environment has NumPy but not SoundFile. The CPU completion validator was corrected to use the standard-library PCM WAV reader; the pinned QA environment remains responsible for model scoring.
- A preliminary partial validation therefore reported `media_read: ModuleNotFoundError(soundfile)`, and the next draft reported `codes_content` for valid `[1,T,16]` files. Both were implementation failures, not dataset failures, and were corrected before the successful zero-media-error audit above.
- The first corruption-probe import used a file-spec loader without adding `training-data/` to `sys.path` and failed on the local `synthesize_vivos` import. The corrected probe added the repository script directory and completed; no data artifact was involved in the import failure.

## Review corrections

A read-only code review found and corrected four pre-runtime state-boundary defects: media errors were initially scored but blocked from retry selection; resumed metric gates were initially trusted rather than recomputed; rejected candidates were incorrectly required to receive a `pass` listening label; and nonterminal successful commands could be deduplicated. It also tightened validation-GO hashing, the exact uint32 `[1,T,16]` code contract, terminal decision recomputation, per-attempt temperature provenance, and state-aware command fingerprints. No generated artifact was changed.

The existing `cache_vivos_full.py` consumes the older scalar-v3 final schema and is intentionally not advertised by the v6 postprocess runner. A separate cache compatibility phase remains required after v6 finalization; no cache-readiness claim is made here.

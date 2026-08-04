# Retry-v6 cache and release implementation

Date: 2026-08-04. This is implementation evidence, not a cache or publication
result. No Mimi model, Hugging Face token, network operation, cache build, or
upload was run in this phase.

`finetune/vivos_v6_provenance.py` now owns the shared CPU-only boundary between
the retry-v6 finalizer, the PyTorch-Mimi cache builder, and the immutable release
workflow. It validates the exact production/source plan and attestations; policy
and validation GO; executed and absent rounds 0/1/2; group, candidate, root
generation, retry, QA, selection, accepted/rejected, WAV/code, source-audit, and
manual-evidence hashes; synthesis/model/package/script contracts; and the
executable row-owned RNG formula recorded by the dated erratum. It does not
fabricate scalar-v3 generation sidecars.

The live CPU preflight returned `incomplete` with exit 3, not `invalid`: the two
snapshots observed 1,518 rows / 193 groups and 1,526 rows / 194 groups while
generation continued. The completed historical v6 validation parsed both
attempts and their QA bindings (64 attempt-0 rows, 21 retry-1 rows), reproduced
the frozen GO provenance, and explicitly returned `cache_ready: false` because
that experiment is not the 10,950-row production scope.

One shell wrapper around the first live preflight used `status` as a zsh local
name and failed after the Python process because `status` is read-only. The
preflight itself had already returned the expected JSON. The corrected wrapper
used `preflight_rc` and confirmed exit 3. This was a command-recording error, not
a data or validator failure.

After final QA reaches GO, run the following with the actual final directory and
manual evidence already bound by `aggregate_report.json`:

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
ROOT=/Volumes/data/datasets/hibiki_vi_v2
PLAN=$ROOT/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan.json
FINAL=$ROOT/qa/vivos_qwen3_tts_mlx_retry_v6_full/final
CACHE=$ROOT/cache/vivos_qwen3_tts_mlx_retry_v6_mimi_v2

$PY finetune/cache_vivos_full.py preflight "$PLAN" \
  --accepted "$FINAL/accepted.jsonl" \
  --selection "$FINAL/selection.jsonl" \
  --qa-report "$FINAL/aggregate_report.json"

$PY finetune/cache_vivos_full.py build "$PLAN" \
  --accepted "$FINAL/accepted.jsonl" \
  --selection "$FINAL/selection.jsonl" \
  --qa-report "$FINAL/aggregate_report.json" \
  --dataset-root "$ROOT" \
  --gender-files \
    "$ROOT/raw/vivos/corpus/vivos/train/genders.txt" \
    "$ROOT/raw/vivos/corpus/vivos/test/genders.txt" \
  --out-root "$CACHE" --device mps

$PY finetune/release_vivos_cache.py preflight "$PLAN" \
  --accepted "$FINAL/accepted.jsonl" \
  --selection "$FINAL/selection.jsonl" \
  --qa-report "$FINAL/aggregate_report.json" \
  --cache-root "$CACHE"

$PY finetune/release_vivos_cache.py prepare "$PLAN" \
  --accepted "$FINAL/accepted.jsonl" \
  --selection "$FINAL/selection.jsonl" \
  --qa-report "$FINAL/aggregate_report.json" \
  --cache-root "$CACHE"

$PY finetune/release_vivos_cache.py publish
```

The fixed release directory is
`/Volumes/data/datasets/hibiki_vi_v2/releases/v2/vivos_qwen3_tts_mlx_retry_v6_full/`;
the fixed Hub dataset prefix is
`v2/vivos_qwen3_tts_mlx_retry_v6_full/` in
`huybik/hibiki-zero-vi-full-sft`.

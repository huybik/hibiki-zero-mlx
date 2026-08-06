# VIVOS postprocess runtime repair and resume

Date: 2026-08-06 (Asia/Ho_Chi_Minh)

## Failure

Attempt-0 QA resumed from 8,993 immutable row records and completed all
10,950 rows at 10:09 ICT. The round-0 selector then exited at 10:11 ICT before
creating `selection_round0.json`. The guarded supervisor and unattended release
worker both entered `halted`; no retry, Mimi cache, release, or Hub upload
started.

The selector log records
`importlib.metadata.PackageNotFoundError: soundfile`. It was launched with the
conda base interpreter while QA was scored in the pinned QA environment. A
pre-repair runtime audit also found two additional mismatches hidden behind the
first exception:

| Package | Conda base before repair | QA runtime |
| --- | ---: | ---: |
| `soundfile` | absent | 0.14.0 |
| `transformers` | 5.10.2 | 4.57.3 |
| `numpy` | 2.2.6 | 2.5.1 |
| `scipy` | 1.16.2 | 1.16.2 |
| `torch` | 2.13.0 | 2.13.0 |

This was a postprocess environment-wiring failure, not a generation or QA data
failure. The failed command, return code, log hash, state transitions, and all
completed QA artifacts remain immutable under `supervisor_exclude1/` and
`unattended_release_exclude2/`.

## Repair

At approximately 15:00 ICT, the base runtime was aligned to the exact runtime
recorded in `attempt0_t08/qa_report.json`:

```text
/opt/homebrew/Caskroom/miniconda/base/bin/python -m pip install \
  'soundfile==0.14.0' 'transformers==4.57.3' 'numpy==2.5.1'
```

An import and metadata audit then returned the exact required contract:
Transformers 4.57.3, SciPy 1.16.2, Torch 2.13.0, SoundFile 0.14.0, and NumPy
2.5.1.

`pip check` continues to report repository-wide declared-version conflicts for
the exploratory MLX/Moshi stack, including MLX-LM's Transformers declaration
and Moshi/OpenCV NumPy declarations. These are recorded rather than hidden.
Qwen production generation uses the isolated
`/Volumes/data/envs/hibiki-vivos-mlx-0.4.7` environment; this repair targets the
base-interpreter selector/finalizer/cache boundary only.

A later publisher preflight found that this first repair would have caused a
different downstream failure: Transformers 4.57.3 forced
`huggingface-hub` 0.36.2, while immutable publication requires 1.21.0, and the
base MLX stack expects Transformers 5.x and NumPy below 2.3. Before any retry,
cache, release, or upload, base was restored to Transformers 5.10.2, NumPy
2.2.6, and `huggingface-hub` 1.21.0 while retaining SoundFile 0.14.0. The
supervisor boundary was corrected so selector/finalizer use the pinned QA
environment and validator/cache/release use base. Both installation attempts
and their dependency-conflict output are retained as failed/corrective steps.

## Resume policy

The halted work directories are preserved. Resume uses fresh supervisor and
release-worker directories bound to the same repaired production plan, the
same three speaker exclusions, the current repository commit, and the existing
10,950 immutable QA rows. The new supervisor may reuse successfully attested
generation and QA artifacts but must rerun round-0 selection. Cache and upload
remain conditional on terminal machine GO.

## Terminal no-retry decision

During the resumed selector, the user explicitly changed the execution policy:
do not continue retrying; drop targets that do not pass validation and proceed
with the remaining phases. The supervisor was stopped before retry generation.
The interrupted selector produced no artifact. A standalone, hash-validating
round-0 selection then recorded 8,254 row-gate-passing targets and 2,696
rejections after speaker exclusions, but the passing subset's aggregate WER
was 0.098133, above the frozen 0.08 corpus gate.

The gate is retained. Terminal/no-retry selection deterministically removes the
smallest number of additional passing rows needed to satisfy it: candidates are
ranked by descending `word_errors - 0.08 * reference_words`, with row id as the
tie-break. This removes 497 rows and leaves 7,757 accepted rows with aggregate
WER 0.079983. The fixed policy, pruned-row count, and SHA-256 of the sorted
pruned ids are bound into selection, finalization, independent cache
provenance validation, release metadata, and the dataset card. Retry rounds 1
and 2 remain unexecuted.

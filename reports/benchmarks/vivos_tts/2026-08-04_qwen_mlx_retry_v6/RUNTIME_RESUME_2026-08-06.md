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

## Resume policy

The halted work directories are preserved. Resume uses fresh supervisor and
release-worker directories bound to the same repaired production plan, the
same three speaker exclusions, the current repository commit, and the existing
10,950 immutable QA rows. The new supervisor may reuse successfully attested
generation and QA artifacts but must rerun round-0 selection. Cache and upload
remain conditional on terminal machine GO.

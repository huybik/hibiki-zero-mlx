# VIVOS production speaker exclusion

Date: 2026-08-05 (Asia/Ho_Chi_Minh)

## Decision

Exclude `VIVOSSPK07`, `VIVOSSPK11`, and `VIVOSSPK18` at the terminal
selection boundary. The generated attempt-0 media and QA remain immutable
research evidence, but these speakers cannot enter retries, the accepted
manifest, Mimi cache, or published training shards.

The production plan remains the original 10,950-row provenance envelope. The
release exclusion covers 753 train rows (240 + 242 + 271), leaving at most
10,197 rows and 43 speakers before ordinary row-level QA rejection. Dev and
test speaker scopes are unchanged.

## Evidence and rationale

At the 8,415-row QA snapshot, 6,296 rows (74.82%) passed every row gate and
2,119 (25.18%) failed at least one. Catastrophic ASR WER above 2 affected 76
rows (0.90%); the deliberately shared failure samples therefore overstated the
typical severity. Failures were nevertheless strongly speaker-correlated:

- `VIVOSSPK18`: 256/271 failed (94.5%).
- `VIVOSSPK11`: 157/242 failed (64.9%).
- `VIVOSSPK07`: 151/240 failed (62.9%).

The three speakers produced 26.6% of failures in that snapshot. A single
Vietnamese reference prompt conditions every English target for a speaker, so
this pattern is consistent with speaker/reference-specific cross-lingual
conditioning instability. Exclusion is safer and cheaper than repeatedly
sampling the same conditioning.

## Operational record

At 05:36 ICT, the old unattended cache/upload worker was interrupted before it
created a waiver, cache, release directory, or Hub upload. The original
postprocess supervisor was then stopped to prevent retries for excluded
speakers. Its foreground QA child also terminated, contrary to the intended
parent-only stop. This interruption lost no completed metrics: 8,713 immutable
row files existed at the first check and 8,745 at termination. The replacement
supervisor resumes those files and scores only the remaining rows.

The selector now accepts repeatable `--exclude-speaker` arguments, binds the
sorted exclusion set and row count into its selection-policy hash, prevents
excluded rows from entering retry manifests, and emits explicit row-level
exclusion records. Finalization, provenance validation, cache validation,
release metadata, and the dataset card independently preserve and verify this
scope. Failed media remain packaged as attempt/QA provenance metadata but are
not training samples.

## Pause at 05:53 ICT

The user requested a full pause. The replacement QA had resumed 248 additional
rows, bringing the immutable attempt-0 metric total to 8,993/10,950, but its
supervisor and QA process were no longer live when the pause audit began. The
command history has a spawned record and no completed record for this scoring
invocation; no cause is asserted without evidence. The remaining unattended
release waiter was explicitly stopped. At the terminal check there were no
generation, QA, retry, cache, release, or upload processes; Mimi cache and
release directories did not exist. Resume must audit the interrupted scoring
record and continue from the 8,993 immutable row files.

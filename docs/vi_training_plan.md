# Vietnamese adaptation strategy

This document has been superseded by three implementation plans:

- [Data generation plan](data_generation_plan.md) - real-source acquisition,
  long-form alignment, provenance, QA, and Mimi-cache release gates.
- [Training plan](training_plan.md) - prefix-PAD A/B, mixed-data full SFT,
  cold long-form continuation, and conditional free-running optimization.
- [Validation plan](validation_plan.md) - teacher-forced diagnostics,
  free-running collapse gates, generated-audio scoring, latency, and retention.

The old plan predated the completed 1,114-hour cache and both full-SFT runs. Its
small-data diagnosis and proposed 50-150-hour synthetic target are obsolete.
Historical evidence remains in [the phase-2 post-mortem](phase2_postmortem.md)
and [CONTEXT.md](../CONTEXT.md).

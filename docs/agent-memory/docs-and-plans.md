# Docs and plans

Read when choosing the next research step, editing a plan, or rebuilding a report.
Back to [CONTEXT.md](../../CONTEXT.md) · [Roadmap](#roadmap) · [Evidence](#evidence) ·
[Superseded plans](#superseded-plans) · [Deliverables](#deliverables)

## Roadmap

- [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md) is the active VI→EN roadmap. Phase 1
  turns scripted English corpora into Vietnamese-source rows (Gemini translation, then
  voice-transfer TTS) toward ~40k h. Phase 2 is SFT with per-domain validation, Phase 3 a
  voice-matched fine-tune, and Phase 4 GRPO. Phases run strictly in order.
- [docs/grpo_plan.md](../grpo_plan.md) details Phase 4. No GRPO code exists; build it from that
  plan once SFT passes its promotion gate.
- [docs/mobile_student_plan.md](../mobile_student_plan.md) is the on-device track
  ([mobile-student](mobile-student.md)).
- [docs/finetune.md](../finetune.md), [docs/training_plan.md](../training_plan.md) and
  [docs/validation_plan.md](../validation_plan.md) own the current SFT pod handoff, recipe and
  validation ([sft-training](sft-training.md)).

## Evidence

`docs/analysis/` analyses the run `grounded_v2_full_direct_voice_5epoch`:

- `loss_function.md` covers the objective.
- `validation_collapse_analysis.md` covers the FLEURS validation divergence.
- `step135k-check.md` covers free-running PhoMT holdout examples at step 135k.

The parity plan's diagnosis is built on these documents.

## Superseded plans

These describe earlier repository states and are not instructions:

- `docs/refactor_plan.md`
- `docs/mobile_realtime_vi_en_plan.md` (pinned to commit `74f1b0f`)
- `docs/vision/reports/*.md` (M4 Pro benchmarks and projected, unmeasured iPhone budgets)

## Deliverables

- [docs/technical/report.md](../technical/report.md) is the technical report source.
  `docs/technical/build_pdf.py` (Pandoc, ReportLab, headless Chrome) writes the tracked
  `output/pdf/hibiki-zero-mlx-technical-report.pdf`.
- `docs/vision/vision.md` builds to `docs/vision/vision.pdf` with `docs/vision/build_pdf.sh`
  (Pandoc plus headless Chrome).
- The `1V_P2_TechProposal_Template.*` and `Connect_TechProposal_Submission.*` files are
  proposal templates and submissions, kept as binaries.

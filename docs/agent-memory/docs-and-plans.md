# Docs and plans

Read when choosing the next research step, editing a plan, or rebuilding a report.
Back to [CONTEXT.md](../../CONTEXT.md) · [Roadmap](#roadmap) · [Evidence](#evidence) ·
[Deliverables](#deliverables)

## Roadmap

- [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md) is the active VI→EN roadmap. Phase 1
  turns scripted English corpora into Vietnamese-source rows (Claude Sonnet 5.5 translation,
  then voice-transfer TTS) toward ~40k h. Phase 2 is SFT with per-domain validation, Phase 3 a
  voice-matched fine-tune, and Phase 4 GRPO. Phases run strictly in order.
- [docs/grpo_plan.md](../grpo_plan.md) details Phase 4. No GRPO code exists; build it from that
  plan once SFT passes its promotion gate.
- The roadmap and GRPO plan still describe a PhoMT + FLEURS training mixture and validation.
  FLEURS and the PhoMT code are removed; the training pool is `en_speech`
  ([en-speech-corpus](en-speech-corpus.md)) plus the restored PhoMT cache
  ([sft-data-caches](sft-data-caches.md#phomt-cache)), with no PhoMT holdout. Its list of
  candidate English corpora is also outdated: the review of other corpora is in
  [en-speech-corpus](en-speech-corpus.md#rules).
- [docs/finetune.md](../finetune.md) is the H100 pod guide for the generic trainer
  ([sft-training](sft-training.md)).

## Evidence

`docs/analysis/` analyses the run `grounded_v2_full_direct_voice_5epoch`:

- `loss_function.md` covers the objective.
- `validation_collapse_analysis.md` covers the FLEURS validation divergence.
- `step135k-check.md` covers free-running PhoMT holdout examples at step 135k.

The parity plan's diagnosis is built on these documents.

`docs/vision/reports/*.md` are historical M4 Pro benchmarks and projected, unmeasured iPhone
budgets cited by the report; they are not instructions.

## Deliverables

- [docs/technical/report.md](../technical/report.md) is the technical report source.
  `docs/technical/build_pdf.py` (Pandoc, ReportLab, headless Chrome) writes the gitignored
  `output/pdf/hibiki-zero-mlx-technical-report.pdf`.
- `docs/vision/vision.md` builds to the gitignored `docs/vision/vision.pdf` with
  `docs/vision/build_pdf.sh` (Pandoc plus headless Chrome).
- No built PDFs or other binary deliverables are tracked.

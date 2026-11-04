# Docs and plans

Read when choosing the next research step, editing a plan, or rebuilding a report.
Back to [CONTEXT.md](../../CONTEXT.md) · [Roadmap](#roadmap) · [Evidence](#evidence) ·
[Deliverables](#deliverables)

## Roadmap

- [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md) is the active VI→EN plan (rewritten
  2026-11-01): Hibiki-Zero 3B from training to the phone, SFT on multi-sentence streams with
  Kyutai's loss weights, real-Vietnamese evaluation sets, GRPO, then the phone model. The Swift
  port and the A15 device test are done ([swift-runtime](swift-runtime.md)): the 3B runs at
  0.4x real time sustained there, so the plan takes the shrink branch. It records the decisions
  with their reasons and the timing the cache teaches.
- The plan's stage 7 keeps EN→VI as a possible step once the VI→EN phone model works (user,
  2026-11-03). The tokenizer blocks it first: Vietnamese text needs ~10 tokens/s of speech
  against the text stream's 12.5.
- [docs/grpo_plan.md](../grpo_plan.md) details the GRPO stage, checked against the paper on
  2026-11-04. It is built next, from the second run's chosen checkpoint, and each phase is
  profiled on the H100 before the next; the Mac only checks the reward math on its CPU.
- [docs/finetune.md](../finetune.md) is the H100 pod guide for the generic trainer
  ([sft-training](sft-training.md)).

## Evidence

`docs/analysis/` analyses the run `grounded_v2_full_direct_voice_5epoch`:

- `loss_function.md` covers the objective.
- `validation_collapse_analysis.md` covers the FLEURS validation divergence.
- `step135k-check.md` covers free-running PhoMT holdout examples at step 135k.

The parity plan's diagnosis is built on these documents. `eos_and_lag.md` covers the stream run
`en_speech_streams_epoch1b`. Most of its missing EOS came from evaluating at text temperature 0.4
with an 8 s tail; a third of FLEURS rows still never end. It also records the lag and the
TTS-against-real-speech gap that remain.

`docs/vision/reports/*.md` are historical M4 Pro benchmarks and projected, unmeasured iPhone
budgets cited by the report; they are not instructions.

## Deliverables

- [docs/technical/report.md](../technical/report.md) is the technical report source.
  `docs/technical/build_pdf.py` (Pandoc, ReportLab, headless Chrome) writes the gitignored
  `output/pdf/hibiki-zero-mlx-technical-report.pdf`.
- `docs/vision/vision.md` builds to the gitignored `docs/vision/vision.pdf` with
  `docs/vision/build_pdf.sh` (Pandoc plus headless Chrome).
- No built PDFs or other binary deliverables are tracked.

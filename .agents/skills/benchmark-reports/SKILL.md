---
name: benchmark-reports
description: Use when creating, updating, or archiving benchmark reports for Hibiki-Zero dataset runs, including metrics tables, translation comparison CSVs, and reproducible report artifacts.
---

# Benchmark Reports

Use this skill to run and archive Hibiki-Zero benchmark results. The default
benchmark flow is fresh end-to-end execution:

- a dataset `manifest.csv`
- generated translation text files such as `fr_0000_q4.txt`
- `metrics_all.json` from `remote_dataset/evaluate_translation_text.py`

## Workflow

1. Download a fresh dataset split/limit into a run-specific `remote_dataset/` directory.
2. Run fresh batch inference into a matching run-specific `translations/` directory.
3. Evaluate the generated text files to create `metrics_all.json`.
4. Keep long-term report artifacts under `reports/benchmarks/{dataset_name}/{run_name}/`.
5. Generate a concise Markdown metrics report.
6. Generate a per-sample CSV with source text, reference text, and model output.
7. Keep audio artifacts outside long-term reports; delete `.wav` files only when
   the user asks for cleanup.

Do not reuse old predictions, manifests, or metrics for a benchmark report unless
the user explicitly asks for a derived/subset report from an existing run.

## Report Artifacts

Create these long-term artifacts for each benchmark run:

- Markdown metrics report: `reports/benchmarks/{dataset_name}/{run_name}/metrics.md`
- Translation comparison CSV: `reports/benchmarks/{dataset_name}/{run_name}/translations.csv`

The Markdown report should stay concise: one metrics table with benchmark
context. The CSV should include `sample`, `audio_file`, `duration_s`,
`transcript_fr`, `reference_en`, and the model translation column.

Put subset reports made from existing predictions under
`reports/benchmarks/{dataset_name}/derived/{run_name}/` so they are not confused
with fresh benchmark runs.

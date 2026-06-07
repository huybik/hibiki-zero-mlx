#!/usr/bin/env python
"""Evaluate generated English translation text against a dataset manifest.

This scores the model's generated text sidecars, not the waveform directly. For
speech-to-speech systems this is a practical first pass: compare the emitted text
tokens, or run ASR on generated speech first and score the ASR transcript here.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated translation .txt files against manifest references."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with audio_file and translation_en columns",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="directory containing generated .txt sidecars",
    )
    parser.add_argument(
        "--pattern",
        default="{stem}_q4.txt",
        help="prediction filename pattern; variables: {stem}, {index} (default: {stem}_q4.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N manifest rows",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="optional path to write metrics JSON",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        import sacrebleu
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install sacrebleu\n"
            "or run:\n"
            "  uv run --with sacrebleu python remote_dataset/evaluate_translation_text.py ..."
        ) from exc
    return sacrebleu


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_words = normalize(reference).split()
        hyp_words = normalize(hypothesis).split()
        edits += edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
    return edits / total_words if total_words else 0.0


def read_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if limit is not None:
        rows = rows[:limit]
    return rows


def prediction_path(pred_dir: Path, pattern: str, row: dict[str, str], index: int) -> Path:
    stem = Path(row["audio_file"]).stem
    return pred_dir / pattern.format(stem=stem, index=index)


def main() -> None:
    args = parse_args()
    sacrebleu = require_dependencies()

    rows = read_manifest(args.manifest, args.limit)
    references = []
    hypotheses = []
    missing = []
    scored_files = []

    for index, row in enumerate(rows):
        pred_path = prediction_path(args.pred_dir, args.pattern, row, index)
        if not pred_path.exists():
            missing.append(str(pred_path))
            continue
        references.append(row["translation_en"].strip())
        hypotheses.append(pred_path.read_text(encoding="utf-8").strip())
        scored_files.append(str(pred_path))

    if not hypotheses:
        raise SystemExit("No prediction files found for the requested manifest rows.")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    wer = word_error_rate(references, hypotheses)

    metrics = {
        "num_manifest_rows": len(rows),
        "num_scored": len(hypotheses),
        "num_missing": len(missing),
        "bleu": bleu,
        "chrf": chrf,
        "wer": wer,
        "scored_files": scored_files,
        "missing_files": missing,
    }

    print(f"scored: {metrics['num_scored']} / {metrics['num_manifest_rows']}")
    print(f"BLEU: {bleu:.2f}")
    print(f"chrF: {chrf:.2f}")
    print(f"WER:  {100 * wer:.2f}%")
    if missing:
        print("\nMissing prediction files:")
        for path in missing:
            print(f"  {path}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote: {args.out_json}")


if __name__ == "__main__":
    main()

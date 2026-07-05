#!/usr/bin/env python
"""Batch-translate a manifest's wavs with the q4 fast MLX path.

Loads the LM once (not per file) and writes a `{stem}_{suffix}.txt` sidecar (plus
wav) per row, matching the layout evaluate_translation_text.py expects.
"""
import argparse
import csv
import time
from pathlib import Path

from hibiki_mlx import pipeline as f


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch q4 MLX translation over a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with an audio_file column")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for wav + txt outputs")
    parser.add_argument("--suffix", default="q4", help="output filename suffix (default: q4)")
    parser.add_argument("--tail-s", type=float, default=8.0, help="silence flush seconds (default: 8)")
    parser.add_argument("--limit", type=int, help="process only the first N rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, lm_config, text_tok, _, _ = f.load(f.W)  # load the LM once
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        wav = row["audio_file"]
        stem = Path(wav).stem
        out_wav = args.out_dir / f"{stem}_{args.suffix}.wav"
        out_txt = args.out_dir / f"{stem}_{args.suffix}.txt"
        mimi_enc, mimi_dec = f.make_mimi(f.W, lm_config)  # fresh codec state per file
        f.run(
            wav,
            str(out_wav),
            text_outfile=str(out_txt),
            tail_s=args.tail_s,
            preloaded=(model, lm_config, text_tok, mimi_enc, mimi_dec),
        )
        print(f"[{i + 1}/{len(rows)}] {stem}")
    print(f"\ndone {len(rows)} files in {time.perf_counter() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Score a staged MLX model on evaluation rows as training's free-running evaluation scores a
checkpoint, to gate a quantized bundle against its bf16 one on the Mac. From the repo root:

  python -m inference.tools.gate --model DIR --pairs data/eval/fleurs-vi/dev.jsonl \\
      data/eval/fleurs-en/dev.jsonl [--limit N] [--seed S] --out-dir DIR

Each row's source clip (`vi_audio`, scored against `text_en`, or `en_audio`, against `text_vi`;
the model picks the direction from what it hears) is translated alone in the runtime's file mode
(`pipeline.translate`: the level stage, the source EOS, 20 s of silence, stopping at the first
text EOS after the source EOS, where training's evaluation also stops at an earlier one), sampled as the model's config.json says (`lm_gen_config`), with the MLX seed set to
--seed plus the row's index: give every model compared the same --pairs, --limit and --seed.
Each pairs file is scored as its own set by training's `score_sets` (corpus chrF and BLEU,
length ratio, empty rows, rows that ended and the health gates). Writes <out-dir>/hypotheses.jsonl,
a row each as it finishes, then <out-dir>/scores.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from hibiki_mlx import pipeline
from training.finetune import common
from training.paths import repo_display_path, require_file, resolve_repo_path

FRAME_RATE = 12.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="3b or a staged model directory")
    parser.add_argument(
        "--pairs", type=Path, nargs="+", required=True,
        help="eval.py pairs files (.jsonl), each scored as its own set",
    )
    parser.add_argument("--limit", type=int, help="the first N rows of each pairs file; all by default")
    parser.add_argument("--seed", type=int, default=42, help="row i is sampled from seed + i")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    sets = common.read_eval_sets(args.pairs)
    common.validate_eval_rows(
        [row for set_rows in sets.values() for row in set_rows], "id", duration_column="duration_s"
    )
    rows = [row for set_rows in sets.values() for row in set_rows[: args.limit]]
    sources = [require_file(row[common.SOURCE], "evaluation audio") for row in rows]
    out_dir = resolve_repo_path(args.out_dir)
    hypotheses = out_dir / "hypotheses.jsonl"
    if hypotheses.exists():
        raise FileExistsError(f"Refusing to overwrite {hypotheses}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = pipeline.resolve_weights_dir(args.model)
    model = pipeline.load(model_dir)

    records: list[dict] = []
    with hypotheses.open("w", encoding="utf-8") as handle:
        for index, (row, source) in enumerate(zip(rows, sources, strict=True)):
            mx.random.seed(args.seed + index)
            start = time.perf_counter()
            out = pipeline.translate(model, source, audio=False)
            seconds = time.perf_counter() - start
            # Training's End Offset: its step i heard i - 1 frames of the clip (moshi's LMGen
            # hears a frame of silence first), where the runtime's step t heard t, so the
            # runtime's step t is training's t + 1.
            offset = "" if out.last_word is None else round(
                (out.last_word + 1) / FRAME_RATE - out.duration, 2
            )
            record = {
                "id": row["id"],
                common.EVAL_SET: row[common.EVAL_SET],
                common.SOURCE: row[common.SOURCE],
                common.REFERENCE: row[common.REFERENCE],
                "prediction_text": out.text,
                "eos_found": out.ended,
                "end_offset_s": offset,
                "frames": out.frames,
                "seconds": round(seconds, 2),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] {row['id']} "
                f"{'ended' if out.ended else 'no EOS'} {out.frames} frames {seconds:.1f}s: "
                f"{out.text[:100]}",
                flush=True,
            )

    scores = common.score_sets(records, list(sets))
    for name, metrics in scores.items():
        print(f"{name}: {common.score_summary(metrics)}")
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "model": repo_display_path(model_dir),
                "pairs": [repo_display_path(path) for path in args.pairs],
                "limit": args.limit,
                "seed": args.seed,
                "sets": scores,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {repo_display_path(hypotheses)} and scores.json")


if __name__ == "__main__":
    main()

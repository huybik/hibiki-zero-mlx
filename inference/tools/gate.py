#!/usr/bin/env python
"""Score a staged MLX model on evaluation rows as training's free-running evaluation scores a
checkpoint, to gate a quantized bundle against its bf16 one on the Mac. From the repo root:

  python -m inference.tools.gate --model DIR --pairs data/eval/fleurs-vi/dev.jsonl \\
      data/eval/fleurs-en/dev.jsonl [--limit N] [--seed S] --out-dir DIR

Each row's source clip (`vi_audio`, scored against `text_en`, or `en_audio`, against `text_vi`;
the model picks the direction from what it hears) is translated alone in the runtime's file mode
(`file_mode.translate`: the level stage, the source EOS, 20 s of silence, stopping at the first
text EOS after the source EOS), sampled as the model's config.json says (`lm_gen_config`), with
the MLX seed set to --seed plus the row's index: give every model compared the same --pairs,
--limit and --seed.
Each pairs file is scored as its own set by training's `score_sets` (corpus chrF and BLEU,
length ratio, empty rows, rows that ended, the health gates and the voice scores: the speech's
speaker similarity to the source and its clarity, `training/finetune/voice_scores.py`, on the
Mac's GPU). Training's evaluation also stops
at an EOS written before the source EOS and keeps the text to it, so a row with one scores
differently here; `eos_found` is an EOS after the source EOS. Writes <out-dir>/hypotheses.jsonl,
a row each as it finishes, then <out-dir>/scores.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from hibiki_mlx.file_mode import translate
from hibiki_mlx.model import load, resolve_weights_dir
from training.finetune import common, voice_scores
from training.paths import repo_display_path, require_file, resolve_repo_path

FRAME_RATE = 12.5
SAMPLE_RATE = 24_000


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
    model_dir = resolve_weights_dir(args.model)
    model = load(model_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    records: list[dict] = []
    with hypotheses.open("w", encoding="utf-8") as handle:
        for index, (row, source) in enumerate(zip(rows, sources, strict=True)):
            mx.random.seed(args.seed + index)
            start = time.perf_counter()
            out = translate(model, source)
            seconds = time.perf_counter() - start
            spoken = out.pcm if out.pcm is not None else np.zeros(0, np.float32)
            # Training's End Offset counts a word at the frames of the clip its step heard, plus
            # one: its output i heard i - max_delay + 1 (moshi's LMGen hears max_delay - 1 silent
            # frames first) and it takes max_delay - 2 off. The runtime's step t heard t.
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
                **voice_scores.score(source, spoken, SAMPLE_RATE, device),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] {row['id']} "
                f"{'ended' if out.ended else 'no EOS'} {out.frames} frames {seconds:.1f}s "
                f"speaker_sim {record['speaker_sim']} hnr {record['hnr_db']}: "
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

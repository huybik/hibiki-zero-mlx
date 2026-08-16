#!/usr/bin/env python
"""Materialize a frozen healthy CoVoST2 French-to-English eval control."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

DATASET = "fixie-ai/covost2"
REVISION = "17c8c81e331e7a6929118121771a58c7ef7331d8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the frozen CoVoST2 fr->en control.")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("remote_dataset/covost2_fr_en_control")
    )
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 2:
        raise ValueError("--limit must be at least 2 for paired evaluation")
    if (
        args.min_duration is not None
        and args.max_duration is not None
        and args.min_duration > args.max_duration
    ):
        raise ValueError("--min-duration must be <= --max-duration")

    from datasets import Audio, load_dataset
    import soundfile as sf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET,
        "fr_en",
        split=args.split,
        streaming=True,
        revision=REVISION,
    ).cast_column("audio", Audio(decode=False))
    rows: list[dict[str, str]] = []
    for example in dataset:
        audio = example["audio"]
        if audio.get("bytes") is None:
            raise RuntimeError("Frozen CoVoST2 control row has no embedded audio bytes")
        samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
        duration = len(samples) / sample_rate
        if args.min_duration is not None and duration < args.min_duration:
            continue
        if args.max_duration is not None and duration > args.max_duration:
            continue
        index = len(rows)
        wav_path = args.out_dir / f"fr_{index:04d}.wav"
        if args.overwrite or not wav_path.is_file():
            sf.write(wav_path, samples, sample_rate)
        rows.append(
            {
                "id": f"covost2_{args.split}_{index:04d}",
                "fr_audio": str(wav_path),
                "fr_duration_s": f"{duration:.6f}",
                "text_en": str(example["translation"]).strip(),
            }
        )
        if len(rows) == args.limit:
            break
    if len(rows) != args.limit:
        raise RuntimeError(f"Requested {args.limit} rows but only found {len(rows)}")

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "fr_audio", "fr_duration_s", "text_en"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved frozen {DATASET}@{REVISION} control -> {manifest}")


if __name__ == "__main__":
    main()

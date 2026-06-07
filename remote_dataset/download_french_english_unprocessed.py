#!/usr/bin/env python
"""Download samples from odunola/french-english-unprocessed.

This dataset provides:
  - `audio`: French speech as WAV data
  - `sentence`: French transcript
  - `english_transcript`: English translation
"""
import argparse
import csv
import io
from pathlib import Path
import wave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch French speech samples with French transcript and English text."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="number of samples to download, e.g. 10 or 20 (default: 10)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="dataset split to stream from (default: train)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("remote_dataset") / "french_english_unprocessed_samples",
        help="directory for wav files and manifest.csv",
    )
    parser.add_argument(
        "--dataset",
        default="odunola/french-english-unprocessed",
        help="Hugging Face dataset name (default: odunola/french-english-unprocessed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing wav files instead of leaving them in place",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="maximum audio duration in seconds",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install datasets\n"
            "or:\n"
            "  uv pip install datasets"
        ) from exc
    return Audio, load_dataset


def write_audio(wav_path: Path, audio: dict) -> None:
    if audio.get("bytes") is not None:
        wav_path.write_bytes(audio["bytes"])
        return

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "Audio bytes were not available. Install soundfile for array audio:\n"
            "  pip install soundfile"
        ) from exc
    sf.write(wav_path, audio["array"], audio["sampling_rate"])


def audio_duration_seconds(audio: dict) -> float:
    if audio.get("bytes") is not None:
        with wave.open(io.BytesIO(audio["bytes"]), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return len(audio["array"]) / audio["sampling_rate"]


def duration_matches(
    duration: float,
    min_duration: float | None,
    max_duration: float | None,
) -> bool:
    if min_duration is not None and duration < min_duration:
        return False
    if max_duration is not None and duration > max_duration:
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    if (
        args.min_duration is not None
        and args.max_duration is not None
        and args.min_duration > args.max_duration
    ):
        raise SystemExit("--min-duration must be <= --max-duration")

    Audio, load_dataset = require_dependencies()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    manifest_rows = []
    for example in dataset:
        duration = audio_duration_seconds(example["audio"])
        if not duration_matches(duration, args.min_duration, args.max_duration):
            continue

        index = len(manifest_rows)
        wav_path = args.out_dir / f"fr_{index:04d}.wav"
        if args.overwrite or not wav_path.exists():
            write_audio(wav_path, example["audio"])

        manifest_rows.append(
            {
                "audio_file": str(wav_path),
                "duration_s": f"{duration:.2f}",
                "transcript_fr": example["sentence"].strip(),
                "translation_en": example["english_transcript"].strip(),
            }
        )
        if len(manifest_rows) >= args.limit:
            break

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["audio_file", "duration_s", "transcript_fr", "translation_en"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {len(manifest_rows)} samples in {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

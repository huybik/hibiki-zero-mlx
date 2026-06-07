#!/usr/bin/env python
"""Download fr->en speech-translation samples from fixie-ai/covost2.

This is a partial copy of CoVoST 2 with the Common Voice audio bundled in, so it
loads without a separate `data_dir` (validation/test splits only). Each example
provides:
  - `audio`: French speech (decoded to a numpy array)
  - `sentence`: French transcript
  - `translation`: English reference translation

Audio is decoded and re-written as WAV (Common Voice is mp3), so the files load
cleanly in the inference pipeline.
"""
import argparse
import csv
import io
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CoVoST 2 fr->en samples (audio + FR transcript + EN reference)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="number of samples to download (default: 50)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="dataset split: test or validation (default: test)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("remote_dataset") / "covost2_fr_en_test",
        help="directory for wav files and manifest.csv",
    )
    parser.add_argument(
        "--dataset",
        default="fixie-ai/covost2",
        help="Hugging Face dataset name (default: fixie-ai/covost2)",
    )
    parser.add_argument(
        "--config",
        default="fr_en",
        help="dataset config / language pair (default: fr_en)",
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
            "  pip install -r remote_dataset/requirements.txt"
        ) from exc
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency soundfile. Install with:\n"
            "  pip install soundfile"
        ) from exc
    return Audio, load_dataset, sf


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

    Audio, load_dataset, sf = require_dependencies()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True
    )
    # Decode the raw audio bytes ourselves via soundfile (libsndfile handles
    # wav/flac/mp3) instead of relying on the datasets torchcodec backend.
    dataset = dataset.cast_column("audio", Audio(decode=False))

    manifest_rows = []
    for example in dataset:
        array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
        duration = len(array) / sr
        if not duration_matches(duration, args.min_duration, args.max_duration):
            continue

        index = len(manifest_rows)
        wav_path = args.out_dir / f"fr_{index:04d}.wav"
        if args.overwrite or not wav_path.exists():
            sf.write(str(wav_path), array, sr)

        manifest_rows.append(
            {
                "audio_file": str(wav_path),
                "duration_s": f"{duration:.2f}",
                "transcript_fr": example["sentence"].strip(),
                "translation_en": example["translation"].strip(),
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

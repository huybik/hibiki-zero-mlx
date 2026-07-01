from __future__ import annotations

import csv
import sys
from pathlib import Path

from datasets import Audio, Dataset


# =========================
# Config
# =========================

DATASET_REPO = "anquachdev/PhoMT-en-vi-speech"
PUSH_TO_HUB = True
PRIVATE = False

DATASETS_DIR = Path(r"D:\Code\datasets")
VI_MANIFEST = DATASETS_DIR / "vieNeu" / "outputs" / "vi" / "manifest.csv"
EN_MANIFEST = DATASETS_DIR / "english" / "outputs" / "en" / "manifest.csv"
LOCAL_SAVE_DIR = DATASETS_DIR / "phomt-en-vi-speech"

# Keep pairs with broadly similar lengths. The current generated data has slow
# English outliers, mostly af_nicole at Kokoro speed 1.0.
MIN_DURATION_RATIO = 0.5
MAX_DURATION_RATIO = 1.6


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


def get_audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("soundfile is required to check audio durations.") from error

    info = sf.info(path)
    return info.frames / info.samplerate


def build_rows() -> list[dict]:
    vi_rows = read_manifest(VI_MANIFEST)
    en_rows = read_manifest(EN_MANIFEST)
    shared_indexes = sorted(set(vi_rows) & set(en_rows))

    rows = []
    skipped_missing = 0
    skipped_duration = 0
    for index in shared_indexes:
        vi_row = vi_rows[index]
        en_row = en_rows[index]
        audio_vi = Path(vi_row["audio_path"])
        audio_en = Path(en_row["audio_path"])

        if not audio_vi.exists() or not audio_en.exists():
            skipped_missing += 1
            continue

        en_duration_s = get_audio_duration_seconds(audio_en)
        vi_duration_s = get_audio_duration_seconds(audio_vi)
        duration_ratio = en_duration_s / vi_duration_s if vi_duration_s else 0.0
        if not MIN_DURATION_RATIO <= duration_ratio <= MAX_DURATION_RATIO:
            skipped_duration += 1
            continue

        rows.append(
            {
                "en": en_row["text"],
                "vi": vi_row["text"],
                "audio_en": str(audio_en),
                "audio_vi": str(audio_vi),
                "duration_en_s": en_duration_s,
                "duration_vi_s": vi_duration_s,
                "duration_ratio_en_vi": duration_ratio,
            }
        )

    if skipped_missing or skipped_duration:
        print(
            f"Skipped {skipped_missing} missing-audio pairs and "
            f"{skipped_duration} duration-mismatched pairs "
            f"(allowed EN/VI ratio: {MIN_DURATION_RATIO:g}-{MAX_DURATION_RATIO:g})."
        )

    if not rows:
        raise ValueError(
            "No paired rows found. Generate both English and Vietnamese audio for the same indexes first."
        )

    return rows


def build_dataset() -> Dataset:
    dataset = Dataset.from_list(build_rows())
    dataset = dataset.cast_column("audio_en", Audio(decode=False))
    dataset = dataset.cast_column("audio_vi", Audio(decode=False))
    dataset = dataset.cast_column("audio_en", Audio())
    dataset = dataset.cast_column("audio_vi", Audio())
    return dataset


def main() -> None:
    dataset = build_dataset()
    print(dataset)

    if PUSH_TO_HUB:
        dataset.push_to_hub(DATASET_REPO, private=PRIVATE)
        print(f"Pushed to https://huggingface.co/datasets/{DATASET_REPO}")
    else:
        dataset.save_to_disk(str(LOCAL_SAVE_DIR))
        print(f"Saved local dataset to: {LOCAL_SAVE_DIR}")
        print("Set PUSH_TO_HUB = True to upload.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Upload build failed: {error}", file=sys.stderr)
        raise

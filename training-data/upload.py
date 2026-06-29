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

TRAINING_DATA_DIR = Path(__file__).resolve().parent
VI_MANIFEST = TRAINING_DATA_DIR / "vieNeu" / "outputs" / "vi" / "manifest.csv"
EN_MANIFEST = TRAINING_DATA_DIR / "english" / "outputs" / "en" / "manifest.csv"
LOCAL_SAVE_DIR = TRAINING_DATA_DIR / "phomt-en-vi-speech"


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


def build_rows() -> list[dict]:
    vi_rows = read_manifest(VI_MANIFEST)
    en_rows = read_manifest(EN_MANIFEST)
    shared_indexes = sorted(set(vi_rows) & set(en_rows))

    rows = []
    for index in shared_indexes:
        vi_row = vi_rows[index]
        en_row = en_rows[index]
        audio_vi = Path(vi_row["audio_path"])
        audio_en = Path(en_row["audio_path"])

        if not audio_vi.exists() or not audio_en.exists():
            continue

        rows.append(
            {
                "en": en_row["text"],
                "vi": vi_row["text"],
                "audio_en": str(audio_en),
                "audio_vi": str(audio_vi),
            }
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

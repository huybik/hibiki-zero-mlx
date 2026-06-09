#!/usr/bin/env python
"""Download FLEURS Vietnamese + English and build a parallel vi->en manifest.

FLEURS (google/fleurs) is n-way parallel: every language records the same
FLoRes sentences, so rows in `vi_vn` and `en_us` that share the same `id` are
translations of one another. This script downloads both, writes WAVs, and joins
them on `id` to produce vi->en speech-translation triplets:

  - vi_audio : Vietnamese speech (source)  -> WAV
  - en_audio : English speech (target)     -> WAV (kept for speech->speech)
  - text_vi  : Vietnamese transcript
  - text_en  : English transcript (target text for speech->text)

Note: within a language a sentence id may be recorded by several speakers; we
keep the first occurrence per id on each side, then inner-join on id.
"""
import argparse
import csv
import io
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch FLEURS vi+en and build a parallel manifest.")
    p.add_argument("--split", default="test", help="train | validation | test (default: test)")
    p.add_argument("--limit", type=int, default=0, help="max paired samples (0 = all)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("remote_dataset") / "fleurs_vi_en",
        help="output directory (the split name is appended as a subdirectory)",
    )
    p.add_argument("--overwrite", action="store_true", help="rewrite existing WAVs")
    return p.parse_args()


def require_deps():
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit("Missing 'datasets'. pip install -r remote_dataset/requirements.txt") from exc
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Missing 'soundfile'. pip install soundfile") from exc
    return Audio, load_dataset, sf


def load_side(load_dataset, Audio, sf, cfg: str, split: str):
    """Return {id: (array, sr, transcription)} keeping the first row per id.

    Non-streaming so the parquet/audio is cached to disk (resumable, retried,
    and reused across splits) instead of re-fetched from the network each run.
    """
    ds = load_dataset("google/fleurs", cfg, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = {}
    for ex in ds:
        sid = ex["id"]
        if sid in rows:
            continue
        audio = ex["audio"]
        if audio.get("bytes") is not None:
            array, sr = sf.read(io.BytesIO(audio["bytes"]))
        else:
            array, sr = sf.read(audio["path"])
        rows[sid] = (array, sr, ex["transcription"].strip())
    return rows


def main() -> None:
    args = parse_args()
    Audio, load_dataset, sf = require_deps()
    out_dir = args.out_dir / args.split  # isolate each split in its own subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    vi_dir = out_dir / "vi"
    en_dir = out_dir / "en"
    vi_dir.mkdir(exist_ok=True)
    en_dir.mkdir(exist_ok=True)

    print(f"Loading FLEURS vi_vn [{args.split}] ...")
    vi = load_side(load_dataset, Audio, sf, "vi_vn", args.split)
    print(f"  {len(vi)} Vietnamese sentences")
    print(f"Loading FLEURS en_us [{args.split}] ...")
    en = load_side(load_dataset, Audio, sf, "en_us", args.split)
    print(f"  {len(en)} English sentences")

    common = sorted(set(vi) & set(en))
    print(f"Paired on id: {len(common)} sentences")

    rows = []
    for sid in common:
        vi_arr, vi_sr, vi_txt = vi[sid]
        en_arr, en_sr, en_txt = en[sid]
        vi_wav = vi_dir / f"vi_{sid:05d}.wav"
        en_wav = en_dir / f"en_{sid:05d}.wav"
        if args.overwrite or not vi_wav.exists():
            sf.write(str(vi_wav), vi_arr, vi_sr)
        if args.overwrite or not en_wav.exists():
            sf.write(str(en_wav), en_arr, en_sr)
        rows.append(
            {
                "id": sid,
                "vi_audio": str(vi_wav),
                "en_audio": str(en_wav),
                "vi_duration_s": f"{len(vi_arr) / vi_sr:.2f}",
                "en_duration_s": f"{len(en_arr) / en_sr:.2f}",
                "text_vi": vi_txt,
                "text_en": en_txt,
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "vi_audio",
                "en_audio",
                "vi_duration_s",
                "en_duration_s",
                "text_vi",
                "text_en",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} paired samples in {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

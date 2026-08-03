"""Download, verify, extract, and manifest the pinned VIVOS source corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "AILAB-VNUHCM/vivos"
REVISION = "3cbfb2502e5e84776b4b778b020a09759f723f52"
SOURCE_FILE = "data/vivos.tar.gz"
ARCHIVE_SIZE = 1_474_408_300
ARCHIVE_SHA256 = "147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca"
LICENSE = "CC BY-NC-SA 4.0"
SCHEMA = "hibiki_vi_source_manifest_v1"
VERSION = "vivos_3cbfb250_source_v1"
DEV_SPEAKER_COUNT = 5
DEV_POLICY_SEED = "hibiki-vi-v2-vivos-dev-v1"
DEFAULT_ROOT = Path("/Volumes/data/datasets/hibiki_vi_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned VIVOS corpus and speaker-disjoint source manifests."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    size = path.stat().st_size
    if size != ARCHIVE_SIZE:
        raise RuntimeError(f"Archive size mismatch: expected {ARCHIVE_SIZE}, found {size}: {path}")
    digest = sha256_file(path)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"Archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, found {digest}: {path}"
        )


def safe_extract(archive_path: Path, corpus_dir: Path) -> None:
    marker_path = corpus_dir / ".vivos-source.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("archive_sha256") != ARCHIVE_SHA256:
            raise RuntimeError(f"Extraction marker does not match pinned archive: {marker_path}")
        return
    if corpus_dir.exists():
        raise RuntimeError(f"Refusing to replace unmarked extraction directory: {corpus_dir}")

    corpus_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".vivos-extract-", dir=corpus_dir.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            destination = temp_dir.resolve()
            for member in archive.getmembers():
                target = (temp_dir / member.name).resolve()
                if os.path.commonpath((destination, target)) != str(destination):
                    raise RuntimeError(f"Unsafe archive path: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(f"Unsupported archive member: {member.name}")
            archive.extractall(temp_dir, filter="data")

        marker_path = temp_dir / ".vivos-source.json"
        marker_path.write_text(
            json.dumps(
                {
                    "archive_sha256": ARCHIVE_SHA256,
                    "repo_id": REPO_ID,
                    "revision": REVISION,
                    "source_file": SOURCE_FILE,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_dir.rename(corpus_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def find_one(corpus_dir: Path, split: str, name: str) -> Path:
    matches = [
        path
        for path in corpus_dir.rglob(name)
        if split in path.relative_to(corpus_dir).parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {split}/{name}, found {len(matches)}")
    return matches[0]


def read_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"Malformed prompt at {path}:{line_number}")
        utterance_id, transcript = fields
        if utterance_id in prompts:
            raise RuntimeError(f"Duplicate prompt id {utterance_id} in {path}")
        prompts[utterance_id] = transcript
    return prompts


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def select_dev_speakers(train_speakers: set[str]) -> list[str]:
    def key(speaker: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{DEV_POLICY_SEED}\0{speaker}".encode()).hexdigest()
        return digest, speaker

    return sorted(train_speakers, key=key)[:DEV_SPEAKER_COUNT]


def load_rows(corpus_dir: Path) -> list[dict[str, object]]:
    split_data: dict[str, tuple[dict[str, str], list[Path]]] = {}
    for split in ("train", "test"):
        prompts = read_prompts(find_one(corpus_dir, split, "prompts.txt"))
        waves_dir = find_one(corpus_dir, split, "waves")
        wavs = sorted(waves_dir.rglob("*.wav"))
        split_data[split] = prompts, wavs

    train_speakers = {path.parent.name for path in split_data["train"][1]}
    dev_speakers = set(select_dev_speakers(train_speakers))
    rows: list[dict[str, object]] = []
    seen_utterances: set[str] = set()

    for official_split, (prompts, wavs) in split_data.items():
        wav_ids = {path.stem for path in wavs}
        missing_audio = sorted(set(prompts) - wav_ids)
        missing_prompt = sorted(wav_ids - set(prompts))
        if missing_audio or missing_prompt:
            raise RuntimeError(
                f"{official_split} prompt/audio mismatch: "
                f"missing_audio={missing_audio[:5]}, missing_prompt={missing_prompt[:5]}"
            )

        for audio_path in wavs:
            utterance_id = audio_path.stem
            if utterance_id in seen_utterances:
                raise RuntimeError(f"Duplicate utterance id across official splits: {utterance_id}")
            seen_utterances.add(utterance_id)
            speaker_id = audio_path.parent.name
            eligibility_split = (
                "test"
                if official_split == "test"
                else "dev"
                if speaker_id in dev_speakers
                else "train"
            )
            with wave.open(str(audio_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                num_frames = wav.getnframes()
                compression = wav.getcomptype()
            if compression != "NONE":
                raise RuntimeError(f"Expected PCM WAV, found {compression}: {audio_path}")

            transcript = prompts[utterance_id]
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "id": f"vivos:{official_split}:{utterance_id}",
                    "audio_path": str(audio_path.resolve()),
                    "text_vi": transcript,
                    "normalized_text_sha256": hashlib.sha256(
                        normalize_text(transcript).encode("utf-8")
                    ).hexdigest(),
                    "official_split": official_split,
                    "eligibility_split": eligibility_split,
                    "speaker_id": speaker_id,
                    "duration_s": round(num_frames / sample_rate, 6),
                    "num_frames": num_frames,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "sample_width_bytes": sample_width,
                    "corpus": "VIVOS",
                    "corpus_revision": REVISION,
                    "license": LICENSE,
                    "source_repo": REPO_ID,
                    "source_file": SOURCE_FILE,
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "audio_sha256": sha256_file(audio_path),
                }
            )
    rows = sorted(rows, key=lambda row: str(row["id"]))

    # Evaluation has priority: preserve official test, then the selected dev
    # speakers, and exclude conflicting official-train rows from optimization.
    test_hashes = {
        str(row["normalized_text_sha256"])
        for row in rows
        if row["eligibility_split"] == "test"
    }
    for row in rows:
        if row["eligibility_split"] == "dev" and row["normalized_text_sha256"] in test_hashes:
            row["eligibility_split"] = "excluded_text_overlap"
            row["exclusion_reason"] = "normalized VI transcript overlaps official test"
    dev_hashes = {
        str(row["normalized_text_sha256"])
        for row in rows
        if row["eligibility_split"] == "dev"
    }
    protected_hashes = test_hashes | dev_hashes
    for row in rows:
        if row["eligibility_split"] == "train" and row["normalized_text_sha256"] in protected_hashes:
            row["eligibility_split"] = "excluded_text_overlap"
            row["exclusion_reason"] = "normalized VI transcript overlaps dev or official test"
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def split_overlap(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[str(row["eligibility_split"])].add(str(row[field]))
    result: dict[str, object] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = sorted(values[left] & values[right])
        result[f"{left}_{right}"] = {"count": len(overlap), "examples": overlap[:20]}
    return result


def summarize(rows: list[dict[str, object]], split_field: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in sorted({str(row[split_field]) for row in rows}):
        selected = [row for row in rows if row[split_field] == split]
        summary[split] = {
            "rows": len(selected),
            "hours": round(sum(float(row["duration_s"]) for row in selected) / 3600, 6),
            "speakers": len({str(row["speaker_id"]) for row in selected}),
        }
    return summary


def build_audit(
    rows: list[dict[str, object]], manifest_paths: dict[str, Path], corpus_dir: Path
) -> dict[str, object]:
    ids = [str(row["id"]) for row in rows]
    missing_paths = [str(row["audio_path"]) for row in rows if not Path(str(row["audio_path"])).is_file()]
    outside_paths = [
        str(row["audio_path"])
        for row in rows
        if not Path(str(row["audio_path"])).is_relative_to(corpus_dir.resolve())
    ]
    speaker_overlap = split_overlap(rows, "speaker_id")
    text_overlap = split_overlap(rows, "normalized_text_sha256")
    checks = {
        "duplicate_ids": len(ids) - len(set(ids)),
        "missing_audio_paths": len(missing_paths),
        "paths_outside_corpus": len(outside_paths),
        "speaker_overlap_rows": sum(int(value["count"]) for value in speaker_overlap.values()),
        "normalized_text_overlap_rows": sum(int(value["count"]) for value in text_overlap.values()),
        "official_test_used_for_training_or_dev": sum(
            row["official_split"] == "test" and row["eligibility_split"] != "test"
            for row in rows
        ),
        "official_train_used_for_test": sum(
            row["official_split"] == "train" and row["eligibility_split"] == "test"
            for row in rows
        ),
    }
    failed = {name: count for name, count in checks.items() if count}
    if failed:
        raise RuntimeError(f"VIVOS audit failed: {failed}")

    selected_dev = sorted(
        {str(row["speaker_id"]) for row in rows if row["eligibility_split"] == "dev"}
    )
    return {
        "schema_version": SCHEMA,
        "dataset_version": VERSION,
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "file": SOURCE_FILE,
            "archive_size": ARCHIVE_SIZE,
            "archive_sha256": ARCHIVE_SHA256,
            "license": LICENSE,
        },
        "dev_policy": {
            "eligible_pool": "official train speakers only",
            "seed": DEV_POLICY_SEED,
            "selection": (
                f"Sort official-train speakers by SHA-256(seed + NUL + speaker_id); "
                f"take the first {DEV_SPEAKER_COUNT}."
            ),
            "selected_speakers": selected_dev,
            "text_overlap_policy": (
                "Preserve official test, then accepted dev; mark conflicting official-train "
                "rows excluded_text_overlap so no train/dev/test transcript hash overlaps."
            ),
        },
        "official_split_summary": summarize(rows, "official_split"),
        "eligibility_split_summary": summarize(rows, "eligibility_split"),
        "overlap": {"speaker": speaker_overlap, "normalized_text": text_overlap},
        "checks": checks,
        "manifest_sha256": {
            name: sha256_file(path) for name, path in sorted(manifest_paths.items())
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    raw_dir = root / "raw" / "vivos"
    corpus_dir = raw_dir / "corpus"
    manifests_dir = root / "manifests"
    audits_dir = root / "audits"
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=SOURCE_FILE,
            repo_type="dataset",
            revision=REVISION,
            local_dir=raw_dir,
        )
    )
    verify_archive(archive_path)
    print(f"Verified archive: {archive_path}", flush=True)
    safe_extract(archive_path, corpus_dir)
    print(f"Prepared corpus: {corpus_dir}", flush=True)

    rows = load_rows(corpus_dir)
    split_rows = {
        "source": rows,
        "train": [row for row in rows if row["eligibility_split"] == "train"],
        "dev": [row for row in rows if row["eligibility_split"] == "dev"],
        "test": [row for row in rows if row["eligibility_split"] == "test"],
        "excluded": [
            row for row in rows if str(row["eligibility_split"]).startswith("excluded_")
        ],
    }
    manifest_paths = {
        name: manifests_dir / f"{VERSION}_{name}.jsonl" for name in split_rows
    }
    for name, selected in split_rows.items():
        write_jsonl(manifest_paths[name], selected)

    audit = build_audit(rows, manifest_paths, corpus_dir)
    audit_path = audits_dir / f"{VERSION}_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split, summary in audit["eligibility_split_summary"].items():
        print(
            f"{split}: {summary['rows']} rows, {summary['hours']:.3f} h, "
            f"{summary['speakers']} speakers"
        )
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()

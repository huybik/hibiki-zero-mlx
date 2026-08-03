"""Prepare and run the pinned VIVOS Qwen3-TTS voice-clone pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

SCHEMA = "hibiki_vivos_qwen3_tts_pilot_v1"
QWEN_PACKAGE_VERSION = "0.1.1"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
KOKORO_PACKAGE_VERSION = "0.9.4"
KOKORO_MODEL_ID = "hexgrad/Kokoro-82M"
KOKORO_MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SEEDS = (20260803, 20260804)
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
}

# The pilot is deliberately explicit: five train and three dev speakers; four
# female and four male; alternating short/long target clips. Test stays sealed.
# Gender comes from the pinned VIVOS genders.txt files, not an inference model.
PILOT_SPEAKERS = (
    {
        "speaker_id": "VIVOSSPK13",
        "gender": "female",
        "split": "train",
        "duration_slice": "short",
        "reference_id": "vivos:train:VIVOSSPK13_184",
        "target_id": "vivos:train:VIVOSSPK13_144",
    },
    {
        "speaker_id": "VIVOSSPK06",
        "gender": "female",
        "split": "train",
        "duration_slice": "long",
        "reference_id": "vivos:train:VIVOSSPK06_R005",
        "target_id": "vivos:train:VIVOSSPK06_T028",
    },
    {
        "speaker_id": "VIVOSSPK26",
        "gender": "female",
        "split": "train",
        "duration_slice": "short",
        "reference_id": "vivos:train:VIVOSSPK26_300",
        "target_id": "vivos:train:VIVOSSPK26_058",
    },
    {
        "speaker_id": "VIVOSSPK42",
        "gender": "female",
        "split": "train",
        "duration_slice": "long",
        "reference_id": "vivos:train:VIVOSSPK42_097",
        "target_id": "vivos:train:VIVOSSPK42_298",
    },
    {
        "speaker_id": "VIVOSSPK08",
        "gender": "male",
        "split": "train",
        "duration_slice": "short",
        "reference_id": "vivos:train:VIVOSSPK08_R006",
        "target_id": "vivos:train:VIVOSSPK08_R116",
    },
    {
        "speaker_id": "VIVOSSPK39",
        "gender": "male",
        "split": "dev",
        "duration_slice": "long",
        "reference_id": "vivos:train:VIVOSSPK39_144",
        "target_id": "vivos:train:VIVOSSPK39_115",
    },
    {
        "speaker_id": "VIVOSSPK37",
        "gender": "male",
        "split": "dev",
        "duration_slice": "short",
        "reference_id": "vivos:train:VIVOSSPK37_064",
        "target_id": "vivos:train:VIVOSSPK37_131",
    },
    {
        "speaker_id": "VIVOSSPK33",
        "gender": "male",
        "split": "dev",
        "duration_slice": "long",
        "reference_id": "vivos:train:VIVOSSPK33_067",
        "target_id": "vivos:train:VIVOSSPK33_021",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Build the immutable pilot plan (standard library only)."
    )
    prepare.add_argument("manifests", type=Path, nargs="+")
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path)
    prepare.add_argument("--kokoro-voice-map", type=Path, required=True)

    generate = subparsers.add_parser(
        "generate", help="Run the prepared plan in a qwen-tts CUDA environment."
    )
    generate.add_argument("plan", type=Path)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument("--dataset-root", type=Path)

    kokoro = subparsers.add_parser(
        "generate-kokoro", help="Generate the pinned matched-Kokoro baseline."
    )
    kokoro.add_argument("plan", type=Path)
    kokoro.add_argument("--device", default="cpu")
    return parser.parse_args()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise RuntimeError(f"Empty JSONL line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def immutable_write(path: Path, value: bytes) -> None:
    if path.exists() and path.read_bytes() != value:
        raise RuntimeError(f"Refusing to change immutable pilot artifact: {path}")
    atomic_write_bytes(path, value)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_bytes(path, "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"))


def git_commit() -> str:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def audio_record(row: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    path = Path(str(row["audio_path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    expected_sha = str(row.get("audio_sha256", ""))
    if actual_sha != expected_sha:
        raise RuntimeError(f"Audio SHA-256 mismatch for {row['id']}: {actual_sha}")
    if not path.is_relative_to(dataset_root):
        raise RuntimeError(f"Audio is outside the dataset root: {path}")
    return {
        "path": str(path),
        "dataset_relative_path": str(path.relative_to(dataset_root)),
        "sha256": actual_sha,
        "duration_s": row["duration_s"],
        "sample_rate_hz": row["sample_rate_hz"],
        "channels": row["channels"],
        "sample_width_bytes": row["sample_width_bytes"],
    }


def translation_record(row: dict[str, Any]) -> dict[str, Any]:
    translation = row.get("translation")
    if not isinstance(translation, dict):
        raise RuntimeError(f"{row.get('id')} is not an accepted translated row")
    required = (
        "schema_version",
        "prompt_version",
        "prompt_sha256",
        "request_entry_sha256",
        "request_file_sha256",
        "input_text_sha256",
        "target_text_sha256",
        "source_manifest_sha256",
        "model_requested",
        "model_version",
        "batch_job_name",
        "finish_reason",
    )
    missing = [field for field in required if not translation.get(field)]
    if missing or translation.get("finish_reason") != "STOP":
        raise RuntimeError(f"Invalid accepted translation for {row['id']}: missing={missing}")
    input_sha = sha256_bytes(str(row["text_vi"]).encode("utf-8"))
    target_sha = sha256_bytes(str(row["text_en"]).encode("utf-8"))
    if (
        translation["input_text_sha256"] != input_sha
        or translation["target_text_sha256"] != target_sha
    ):
        raise RuntimeError(f"Translation text hash mismatch for {row['id']}")
    return {field: translation[field] for field in required}


def kokoro_speed(voice: str) -> float:
    speed_by_voice = {"af_nicole": 1.35, "am_michael": 1.10}
    total = weight_total = 0.0
    for part in voice.split(","):
        name, _, weight_text = part.partition(":")
        weight = float(weight_text) if weight_text else 1.0
        total += speed_by_voice.get(name, 1.0) * weight
        weight_total += weight
    return total / weight_total


def prepare(
    manifests: list[Path],
    out_dir: Path,
    dataset_root: Path | None,
    kokoro_voice_map: Path,
) -> None:
    out_dir = out_dir.expanduser().resolve()
    resolved_manifests = [path.expanduser().resolve() for path in manifests]
    if dataset_root is None:
        inferred_roots = {path.parent.parent for path in resolved_manifests}
        if len(inferred_roots) != 1:
            raise RuntimeError("Cannot infer one dataset root; pass --dataset-root")
        dataset_root = inferred_roots.pop()
    else:
        dataset_root = dataset_root.expanduser().resolve()
    voice_map_path = kokoro_voice_map.expanduser().resolve()
    voice_map_document = json.loads(voice_map_path.read_text(encoding="utf-8"))
    voice_map = voice_map_document.get("map")
    if not isinstance(voice_map, dict):
        raise RuntimeError(f"Missing map object in {voice_map_path}")
    manifest_records: list[dict[str, str]] = []
    rows_by_id: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for path in resolved_manifests:
        record = {"path": str(path), "sha256": sha256_file(path)}
        manifest_records.append(record)
        for row in read_jsonl(path):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in rows_by_id:
                raise RuntimeError(f"Empty or duplicate accepted id: {row_id!r}")
            rows_by_id[row_id] = (row, record)

    plan_rows: list[dict[str, Any]] = []
    for spec in PILOT_SPEAKERS:
        try:
            target, target_manifest = rows_by_id[spec["target_id"]]
            reference, reference_manifest = rows_by_id[spec["reference_id"]]
        except KeyError as error:
            raise RuntimeError(
                f"Pinned pilot id missing from accepted manifests: {error}"
            ) from error
        for role, row in (("target", target), ("reference", reference)):
            if row.get("speaker_id") != spec["speaker_id"]:
                raise RuntimeError(f"Pinned {role} speaker mismatch: {row['id']}")
            if row.get("eligibility_split") != spec["split"]:
                raise RuntimeError(f"Pinned {role} split mismatch: {row['id']}")
        text_en = str(target.get("text_en", "")).strip()
        if not text_en:
            raise RuntimeError(f"Empty accepted English target: {target['id']}")
        voice = str(voice_map.get(spec["speaker_id"], ""))
        expected_prefix = "af_" if spec["gender"] == "female" else "am_"
        if not voice or any(
            not part.partition(":")[0].startswith(expected_prefix) for part in voice.split(",")
        ):
            raise RuntimeError(
                f"Missing or gender-mismatched Kokoro voice for {spec['speaker_id']}"
            )
        target_audio = audio_record(target, dataset_root)
        reference_audio = audio_record(reference, dataset_root)
        provenance = {
            "corpus": target["corpus"],
            "corpus_revision": target["corpus_revision"],
            "license": target["license"],
            "source_repo": target["source_repo"],
            "source_archive_sha256": target["source_archive_sha256"],
            "accepted_manifest": target_manifest,
            "translation": translation_record(target),
        }
        for seed in SEEDS:
            pilot_id = f"{target['id']}|qwen|seed={seed}"
            filename = f"{str(target['id']).replace(':', '_')}.qwen.seed{seed}.wav"
            plan_rows.append(
                {
                    "schema_version": SCHEMA,
                    "pilot_id": pilot_id,
                    "speaker_id": spec["speaker_id"],
                    "gender": spec["gender"],
                    "eligibility_split": spec["split"],
                    "duration_slice": spec["duration_slice"],
                    "target_id": target["id"],
                    "text_vi": target["text_vi"],
                    "text_en": text_en,
                    "source_audio": target_audio,
                    "reference": {
                        "id": reference["id"],
                        "text_vi": reference["text_vi"],
                        "audio": reference_audio,
                        "accepted_manifest": reference_manifest,
                    },
                    "provenance": provenance,
                    "synthesis": {
                        "package": "qwen-tts",
                        "package_version": QWEN_PACKAGE_VERSION,
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "language": "English",
                        "x_vector_only_mode": False,
                        "seed": seed,
                        "generation_config": GENERATION_CONFIG,
                    },
                    "kokoro_baseline": {
                        "package": "kokoro",
                        "package_version": KOKORO_PACKAGE_VERSION,
                        "model_id": KOKORO_MODEL_ID,
                        "model_revision": KOKORO_MODEL_REVISION,
                        "voice_map_sha256": sha256_file(voice_map_path),
                        "voice": voice,
                        "speed": kokoro_speed(voice),
                        "output_wav": str(
                            Path("kokoro_wavs")
                            / f"{str(target['id']).replace(':', '_')}.kokoro.wav"
                        ),
                    },
                    "output_wav": str(Path("wavs") / filename),
                }
            )

    plan_rows.sort(key=lambda row: str(row["pilot_id"]))
    plan_bytes = "".join(canonical_json(row) + "\n" for row in plan_rows).encode("utf-8")
    plan_path = out_dir / "pilot_plan.jsonl"
    config = {
        "schema_version": SCHEMA,
        "repository_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_root_at_prepare": str(dataset_root),
        "accepted_manifests": sorted(manifest_records, key=lambda item: item["path"]),
        "kokoro_voice_map": {
            "path": str(voice_map_path),
            "sha256": sha256_file(voice_map_path),
        },
        "speaker_allowlist": list(PILOT_SPEAKERS),
        "seeds": list(SEEDS),
        "synthesis": {
            "package": "qwen-tts",
            "package_version": QWEN_PACKAGE_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "generation_config": GENERATION_CONFIG,
        },
        "kokoro_baseline": {
            "package": "kokoro",
            "package_version": KOKORO_PACKAGE_VERSION,
            "model_id": KOKORO_MODEL_ID,
            "model_revision": KOKORO_MODEL_REVISION,
        },
        "planned_rows": len(plan_rows),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_bytes(plan_bytes),
    }
    immutable_write(plan_path, plan_bytes)
    immutable_write(
        out_dir / "pilot_config.json",
        (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"Prepared {len(plan_rows)} outputs for {len(PILOT_SPEAKERS)} speakers")
    print(f"Plan: {plan_path}")
    print(f"SHA-256: {config['plan_sha256']}")


def require_package(name: str, expected: str, purpose: str) -> str:
    try:
        installed = package_version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"{purpose} requires {name}=={expected}") from error
    if installed != expected:
        raise RuntimeError(f"{purpose} requires {name}=={expected}, found {installed}")
    return installed


def validate_plan(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(PILOT_SPEAKERS) * len(SEEDS):
        raise RuntimeError(f"Expected 16 pinned pilot rows, found {len(rows)}")
    ids = [str(row.get("pilot_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise RuntimeError("Pilot plan has empty or duplicate ids")
    expected = {
        (
            spec["speaker_id"],
            spec["split"],
            spec["duration_slice"],
            spec["reference_id"],
            spec["target_id"],
            seed,
        )
        for spec in PILOT_SPEAKERS
        for seed in SEEDS
    }
    actual = {
        (
            row.get("speaker_id"),
            row.get("eligibility_split"),
            row.get("duration_slice"),
            row.get("reference", {}).get("id"),
            row.get("target_id"),
            row.get("synthesis", {}).get("seed"),
        )
        for row in rows
    }
    if actual != expected:
        raise RuntimeError("Pilot rows do not match the sealed train/dev allowlist")
    for row in rows:
        if row.get("schema_version") != SCHEMA:
            raise RuntimeError(f"Pilot schema mismatch: {row.get('schema_version')}")
        synthesis = row.get("synthesis", {})
        if (
            synthesis.get("model_id") != MODEL_ID
            or synthesis.get("model_revision") != MODEL_REVISION
            or synthesis.get("package_version") != QWEN_PACKAGE_VERSION
            or synthesis.get("generation_config") != GENERATION_CONFIG
        ):
            raise RuntimeError(f"Unpinned synthesis config in {row['pilot_id']}")
        baseline = row.get("kokoro_baseline", {})
        if (
            baseline.get("package_version") != KOKORO_PACKAGE_VERSION
            or baseline.get("model_id") != KOKORO_MODEL_ID
            or baseline.get("model_revision") != KOKORO_MODEL_REVISION
            or not baseline.get("voice_map_sha256")
            or not baseline.get("voice")
        ):
            raise RuntimeError(f"Unpinned Kokoro baseline in {row['pilot_id']}")


def validate_audio_inputs(
    rows: list[dict[str, Any]], dataset_root: Path | None = None
) -> dict[str, Path]:
    root = dataset_root.expanduser().resolve() if dataset_root else None
    verified: dict[str, str] = {}
    resolved: dict[str, Path] = {}
    for row in rows:
        for audio in (row["source_audio"], row["reference"]["audio"]):
            planned_path = str(audio["path"])
            relative = Path(str(audio["dataset_relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe dataset-relative audio path: {relative}")
            path = root / relative if root is not None else Path(planned_path)
            expected = str(audio["sha256"])
            actual = verified.get(str(path))
            if actual is None:
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual = sha256_file(path)
                verified[str(path)] = actual
            if actual != expected:
                raise RuntimeError(f"Planned input audio hash mismatch: {path}")
            resolved[planned_path] = path
    return resolved


def planned_output_path(row: dict[str, Any], plan_path: Path) -> Path:
    relative = Path(str(row["output_wav"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe planned output path: {relative}")
    return (plan_path.parent / relative).resolve()


def baseline_output_path(row: dict[str, Any], plan_path: Path) -> Path:
    relative = Path(str(row["kokoro_baseline"]["output_wav"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe planned Kokoro output path: {relative}")
    return (plan_path.parent / relative).resolve()


def atomic_write_wav(path: Path, audio: Any, sample_rate: int, soundfile: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        soundfile.write(str(temp_path), audio, sample_rate, subtype="PCM_16")
        with temp_path.open("rb+") as source:
            os.fsync(source.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def generate(plan_path: Path, device: str, dataset_root: Path | None) -> None:
    plan_path = plan_path.expanduser().resolve()
    rows = read_jsonl(plan_path)
    validate_plan(rows)
    input_paths = validate_audio_inputs(rows, dataset_root)
    plan_sha = sha256_file(plan_path)
    config_path = plan_path.parent / "pilot_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("plan_sha256") != plan_sha:
        raise RuntimeError(f"Plan hash does not match {config_path}")
    if not device.startswith("cuda:"):
        raise RuntimeError("Qwen pilot generation is CUDA-only; use --device cuda:N")

    manifest_path = plan_path.parent / "generation.jsonl"
    completed = read_jsonl(manifest_path) if manifest_path.exists() else []
    completed_by_id = {str(row.get("pilot_id", "")): row for row in completed}
    if len(completed_by_id) != len(completed):
        raise RuntimeError(f"Duplicate completed ids in {manifest_path}")
    plan_by_id = {str(row["pilot_id"]): row for row in rows}
    plan_ids = set(plan_by_id)
    if set(completed_by_id) - plan_ids:
        raise RuntimeError(f"Generation manifest contains ids outside the plan: {manifest_path}")
    for item in completed:
        planned = plan_by_id[str(item["pilot_id"])]
        output = planned_output_path(planned, plan_path)
        if (
            item.get("plan_sha256") != plan_sha
            or item.get("output_wav") != str(output)
            or item.get("speaker_id") != planned["speaker_id"]
            or item.get("target_id") != planned["target_id"]
            or item.get("replicate_seed") != planned["synthesis"]["seed"]
            or item.get("synthesis") != planned["synthesis"]
            or sha256_file(output) != item.get("audio_sha256")
        ):
            raise RuntimeError(f"Completed output provenance mismatch: {item['pilot_id']}")

    pending = [row for row in rows if row["pilot_id"] not in completed_by_id]
    if not pending:
        print(f"All {len(rows)} planned outputs already complete: {manifest_path}")
        return
    for row in pending:
        output = planned_output_path(row, plan_path)
        if output.exists():
            raise RuntimeError(f"Unrecorded output exists; refusing to overwrite: {output}")

    qwen_version = require_package("qwen-tts", QWEN_PACKAGE_VERSION, "Qwen generation")
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel
    except ImportError as error:
        raise RuntimeError(
            "Generation requires qwen-tts, torch, huggingface-hub, numpy, and soundfile "
            "in a CUDA environment"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU or MPS fallback is implemented")

    # qwen-tts 0.1.1 does not forward from_pretrained kwargs to its processor.
    # Resolve the pinned snapshot first so model and processor load the same revision.
    model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    model_root = Path(model_path)
    model_snapshot_hashes = {
        str(path.relative_to(model_root)): sha256_file(path)
        for path in sorted(model_root.rglob("*"))
        if path.is_file()
    }
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    prompt_by_speaker: dict[str, Any] = {}
    for number, row in enumerate(pending, start=1):
        speaker = str(row["speaker_id"])
        if speaker not in prompt_by_speaker:
            reference = row["reference"]
            prompt_by_speaker[speaker] = model.create_voice_clone_prompt(
                ref_audio=str(input_paths[reference["audio"]["path"]]),
                ref_text=reference["text_vi"],
                x_vector_only_mode=False,
            )
        seed = int(row["synthesis"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.monotonic()
        wavs, sample_rate = model.generate_voice_clone(
            text=row["text_en"],
            language="English",
            voice_clone_prompt=prompt_by_speaker[speaker],
            **GENERATION_CONFIG,
        )
        if len(wavs) != 1:
            raise RuntimeError(f"Expected one generated waveform for {row['pilot_id']}")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"Generated empty or non-finite audio for {row['pilot_id']}")
        output_path = planned_output_path(row, plan_path)
        atomic_write_wav(output_path, audio, int(sample_rate), sf)
        result = {
            "schema_version": SCHEMA,
            "pilot_id": row["pilot_id"],
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "speaker_id": speaker,
            "target_id": row["target_id"],
            "replicate_seed": seed,
            "output_wav": str(output_path),
            "audio_sha256": sha256_file(output_path),
            "sample_rate_hz": int(sample_rate),
            "num_samples": int(audio.size),
            "duration_s": round(audio.size / int(sample_rate), 6),
            "generation_seconds": round(time.monotonic() - started, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "qwen-tts": qwen_version,
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "soundfile": package_version("soundfile"),
                "device": device,
                "cuda": torch.version.cuda,
            },
            "model_snapshot": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "files_sha256": model_snapshot_hashes,
            },
            "synthesis": row["synthesis"],
        }
        completed_by_id[str(row["pilot_id"])] = result
        atomic_write_jsonl(
            manifest_path,
            [completed_by_id[key] for key in sorted(completed_by_id)],
        )
        print(f"[{number}/{len(pending)}] {row['pilot_id']} -> {output_path}", flush=True)
    print(f"Generation manifest: {manifest_path}")


def generate_kokoro(plan_path: Path, device: str) -> None:
    plan_path = plan_path.expanduser().resolve()
    rows = read_jsonl(plan_path)
    validate_plan(rows)
    plan_sha = sha256_file(plan_path)
    config_path = plan_path.parent / "pilot_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("plan_sha256") != plan_sha:
        raise RuntimeError(f"Plan hash does not match {config_path}")
    if device != "cpu":
        raise RuntimeError("The matched-Kokoro pilot baseline is pinned to CPU")

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_id = str(row["target_id"])
        previous = unique.setdefault(target_id, row)
        if previous["kokoro_baseline"] != row["kokoro_baseline"]:
            raise RuntimeError(f"Kokoro baseline differs across replicates: {target_id}")

    manifest_path = plan_path.parent / "kokoro_generation.jsonl"
    completed = read_jsonl(manifest_path) if manifest_path.exists() else []
    completed_by_id = {str(row.get("target_id", "")): row for row in completed}
    if len(completed_by_id) != len(completed) or set(completed_by_id) - set(unique):
        raise RuntimeError(f"Invalid completed Kokoro manifest: {manifest_path}")
    for target_id, item in completed_by_id.items():
        planned = unique[target_id]
        output = baseline_output_path(planned, plan_path)
        if (
            item.get("plan_sha256") != plan_sha
            or item.get("output_wav") != str(output)
            or item.get("speaker_id") != planned["speaker_id"]
            or item.get("baseline") != planned["kokoro_baseline"]
            or sha256_file(output) != item.get("audio_sha256")
        ):
            raise RuntimeError(f"Completed Kokoro provenance mismatch: {target_id}")

    pending = [row for target_id, row in unique.items() if target_id not in completed_by_id]
    if not pending:
        print(f"All {len(unique)} matched-Kokoro outputs already complete: {manifest_path}")
        return
    for row in pending:
        output = baseline_output_path(row, plan_path)
        if output.exists():
            raise RuntimeError(f"Unrecorded Kokoro output exists; refusing to overwrite: {output}")

    kokoro_version = require_package("kokoro", KOKORO_PACKAGE_VERSION, "Kokoro baseline")
    try:
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from kokoro import KModel, KPipeline
    except ImportError as error:
        raise RuntimeError(
            "Kokoro baseline requires kokoro, torch, huggingface-hub, numpy, and soundfile"
        ) from error

    snapshot = Path(snapshot_download(repo_id=KOKORO_MODEL_ID, revision=KOKORO_MODEL_REVISION))
    config_file = snapshot / "config.json"
    model_file = snapshot / "kokoro-v1_0.pth"
    model = (
        KModel(repo_id=KOKORO_MODEL_ID, config=str(config_file), model=str(model_file))
        .to(device)
        .eval()
    )
    pipeline = KPipeline(lang_code="a", repo_id=KOKORO_MODEL_ID, model=model, device=device)
    voice_cache: dict[str, Any] = {}

    def resolve_voice(voice: str) -> Any:
        if voice not in voice_cache:
            parts = []
            weight_total = 0.0
            for part in voice.split(","):
                name, _, weight_text = part.partition(":")
                weight = float(weight_text) if weight_text else 1.0
                voice_file = snapshot / "voices" / f"{name}.pt"
                parts.append(pipeline.load_single_voice(str(voice_file)) * weight)
                weight_total += weight
            voice_cache[voice] = sum(parts) / weight_total
        return voice_cache[voice]

    for number, row in enumerate(pending, start=1):
        baseline = row["kokoro_baseline"]
        started = time.monotonic()
        parts = [
            result.audio.detach().cpu().numpy()
            for result in pipeline(
                row["text_en"],
                voice=resolve_voice(str(baseline["voice"])),
                speed=float(baseline["speed"]),
            )
            if result.audio is not None
        ]
        audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"Kokoro returned invalid audio for {row['target_id']}")
        output = baseline_output_path(row, plan_path)
        atomic_write_wav(output, audio, 24_000, sf)
        result = {
            "schema_version": SCHEMA,
            "target_id": row["target_id"],
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "speaker_id": row["speaker_id"],
            "output_wav": str(output),
            "audio_sha256": sha256_file(output),
            "sample_rate_hz": 24_000,
            "num_samples": int(audio.size),
            "duration_s": round(audio.size / 24_000, 6),
            "generation_seconds": round(time.monotonic() - started, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "kokoro": kokoro_version,
                "torch": package_version("torch"),
                "soundfile": package_version("soundfile"),
                "device": device,
            },
            "weights": {
                "config_sha256": sha256_file(config_file),
                "model_sha256": sha256_file(model_file),
                "voice_sha256": {
                    name: sha256_file(snapshot / "voices" / f"{name}.pt")
                    for name in sorted(
                        part.partition(":")[0] for part in str(baseline["voice"]).split(",")
                    )
                },
            },
            "baseline": baseline,
        }
        completed_by_id[str(row["target_id"])] = result
        atomic_write_jsonl(manifest_path, [completed_by_id[key] for key in sorted(completed_by_id)])
        print(f"[{number}/{len(pending)}] {row['target_id']} -> {output}", flush=True)
    print(f"Kokoro generation manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare(args.manifests, args.out_dir, args.dataset_root, args.kokoro_voice_map)
    elif args.action == "generate":
        generate(args.plan, args.device, args.dataset_root)
    else:
        generate_kokoro(args.plan, args.device)


if __name__ == "__main__":
    main()

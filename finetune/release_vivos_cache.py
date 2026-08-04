#!/usr/bin/env python
"""Prepare, publish, and clean-room verify the immutable VIVOS cache release."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.cache_vivos_full import (  # noqa: E402
    CACHE_FORMAT,
    audit_shards,
    sha256_bytes,
    sha256_file,
)
from finetune.vivos_v6_provenance import (  # noqa: E402
    IncompleteCampaign,
    provenance_paths,
    validate_finalized,
    validate_historical,
    validate_live_state,
)

SCHEMA = "hibiki_vivos_cache_release_v2"
RELEASE_ID = "vivos_qwen3_tts_mlx_retry_v6_full"
RELEASE_DIR = Path("/Volumes/data/datasets/hibiki_vi_v2/releases/v2") / RELEASE_ID
REMOTE_PREFIX = f"v2/{RELEASE_ID}"
REPO_ID = "huybik/hibiki-zero-vi-full-sft"
HF_HUB_VERSION = "1.21.0"
ZSTANDARD_VERSION = "0.23.0"
TORCH_VERSION = "2.13.0"
LICENSE = "CC BY-NC-SA 4.0"
SOURCE_REPO = "AILAB-VNUHCM/vivos"
SOURCE_REVISION = "3cbfb2502e5e84776b4b778b020a09759f723f52"
SOURCE_ARCHIVE_SHA256 = "147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca"
LOCAL_REPORT = "release_report.json"
EXPECTED_ROWS = 10_950
EXPECTED_SPEAKERS = 46
MIN_LFS_ARCHIVE_BYTES = 10 * 1024 * 1024

PACKAGED_INPUTS = {
    "cache_config": "metadata/cache/cache_config.json",
    "cache_audit": "metadata/cache/cache_audit.json",
    "train_index": "metadata/cache/train_index.csv",
    "dev_index": "metadata/cache/dev_index.csv",
    "source_audit_report": "metadata/source/audit_report.json",
    "source_audit_rows": "metadata/source/row_metrics.jsonl",
    "reference_map": "metadata/source/reference_map.jsonl",
    "reference_report": "metadata/source/reference_map_report.json",
    "production_plan": "metadata/campaign/production_plan.json",
    "production_attestation": "metadata/campaign/production_attestation.json",
    "policy": "metadata/campaign/policy.json",
    "validation_go": "metadata/campaign/validation_go.json",
    "approval": "metadata/campaign/approval_override.json",
    "source_plan": "metadata/campaign/source_plan.jsonl",
    "accepted": "metadata/qa/accepted.jsonl",
    "rejected": "metadata/qa/rejected.jsonl",
    "selection": "metadata/qa/selection.jsonl",
    "selected_candidates": "metadata/qa/selected_candidates.jsonl",
    "qa_report": "metadata/qa/aggregate_report.json",
    "selection_report": "metadata/qa/selection_report.json",
    "manual_required": "metadata/qa/manual_review_required.tsv",
    "manual_evidence": "metadata/qa/manual_evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    def final_inputs(command: argparse.ArgumentParser, *, required: bool) -> None:
        command.add_argument("production_plan", type=Path)
        command.add_argument("--accepted", type=Path, required=required)
        command.add_argument("--selection", type=Path, required=required)
        command.add_argument("--qa-report", type=Path, required=required)

    preflight = commands.add_parser("preflight")
    final_inputs(preflight, required=False)
    preflight.add_argument("--cache-root", type=Path)
    historical = commands.add_parser("preflight-historical")
    historical.add_argument("policy", type=Path)
    historical.add_argument("--qa-root", type=Path, required=True)
    historical.add_argument("--selection-report", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    final_inputs(prepare, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise RuntimeError(f"Empty JSONL line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def attestation(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def producer_attestation(path: Path) -> dict[str, str]:
    record = attestation(path)
    return {"path": record["path"], "sha256": record["sha256"]}


def require_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name != name or not resolved.is_file():
        raise RuntimeError(f"Expected exact {name} artifact: {resolved}")
    return resolved


def require_version(package: str, expected: str) -> str:
    try:
        installed = package_version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(f"Release requires {package}=={expected}") from error
    if installed != expected:
        raise RuntimeError(f"Release requires {package}=={expected}, found {installed}")
    return installed


def immutable_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite release artifact: {destination}")
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())


def compare_audits(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = (
        "schema_version",
        "accepted_rows",
        "cache_rows",
        "missing_ids",
        "unexpected_ids",
        "duplicate_ids",
        "invalid_rows",
        "supervision_totals",
        "complete",
    )
    if {key: actual.get(key) for key in fields} != {key: expected.get(key) for key in fields}:
        raise RuntimeError("Independent cache audit differs from the frozen build audit")


def load_shard_config(torch: Any, shards: list[Path]) -> dict[str, int]:
    config: dict[str, int] | None = None
    for shard in shards:
        payload = torch.load(shard, map_location="cpu")
        shard_config = payload.get("config")
        if not isinstance(shard_config, dict):
            raise RuntimeError(f"Missing shard config: {shard}")
        normalized = {key: int(shard_config[key]) for key in shard_config}
        if config is None:
            config = normalized
        elif normalized != config:
            raise RuntimeError(f"Cache shard config changed: {shard}")
    if config is None:
        raise RuntimeError("Release requires at least one cache shard")
    return config


def validate_indexes(
    torch: Any,
    cache_root: Path,
    indexes: dict[str, Path],
    accepted: list[dict[str, Any]],
) -> None:
    fields = [
        "id",
        "split",
        "speaker_id",
        "gender",
        "stratum",
        "shard",
        "frames",
        "source_frames",
        "target_frames",
        "text_tokens",
        "target_delay_s",
        "target_delay_frames",
        "source_manifest_sha256",
    ]
    for split in ("train", "dev"):
        expected_ids = sorted(
            str(row["id"]) for row in accepted if row["eligibility_split"] == split
        )
        with indexes[split].open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != fields:
                raise RuntimeError(f"Frozen {split} index columns changed")
            rows = list(reader)
        if [row["id"] for row in rows] != expected_ids:
            raise RuntimeError(f"Frozen {split} index scope/order changed")
        samples = {}
        for shard in sorted((cache_root / split).glob("shard_*.pt")):
            payload = torch.load(shard, map_location="cpu")
            for sample in payload["samples"]:
                samples[str(sample["id"])] = (sample, shard.name)
        if len(samples) != len(rows):
            raise RuntimeError(f"Frozen {split} index is not a shard bijection")
        for row in rows:
            sample, shard_name = samples[row["id"]]
            expected = {key: str(sample.get(key, "")) for key in fields if key != "shard"} | {
                "shard": shard_name
            }
            if row != expected:
                raise RuntimeError(f"Frozen index row differs from shard: {row['id']}")


def validate_cache(cache_root: Path, accepted: list[dict[str, Any]]) -> dict[str, Any]:
    require_version("torch", TORCH_VERSION)
    import torch

    accepted_ids = {str(row["id"]) for row in accepted}
    cache_root = cache_root.expanduser().resolve()
    if not cache_root.is_dir():
        raise RuntimeError(f"Cache root is missing: {cache_root}")
    config_path = require_file(cache_root / "cache_config.json", "cache_config.json")
    audit_path = require_file(cache_root / "cache_audit.json", "cache_audit.json")
    config = read_json(config_path)
    audit = read_json(audit_path)
    if config.get("schema_version") != CACHE_FORMAT or audit.get("schema_version") != CACHE_FORMAT:
        raise RuntimeError("Release requires finalized hibiki_vn_lora_cache_v2 artifacts")
    if audit.get("complete") is not True or audit.get("accepted_rows") != len(accepted_ids):
        raise RuntimeError("Cache audit is incomplete or accepted scope changed")
    expected_splits = {
        split: sum(row["eligibility_split"] == split for row in accepted)
        for split in ("train", "dev")
    }
    if config.get("scope") != {
        "rows": len(accepted),
        "splits": expected_splits,
        "test_sealed": True,
    }:
        raise RuntimeError("Cache config scope differs from accepted train/dev rows")
    if audit.get("cache_config") != producer_attestation(config_path):
        raise RuntimeError("Cache config is not bound to the cache audit")
    expected_root = {"cache_config.json", "cache_audit.json", "train", "dev"}
    if {path.name for path in cache_root.iterdir()} != expected_root:
        raise RuntimeError("Cache root contains an unexpected or missing artifact")
    shards = sorted([*cache_root.glob("train/shard_*.pt"), *cache_root.glob("dev/shard_*.pt")])
    recorded = audit.get("shards", [])
    if [producer_attestation(path) for path in shards] != recorded:
        raise RuntimeError("Cache shard set or hash differs from cache_audit.json")
    indexes: dict[str, Path] = {}
    for split in ("train", "dev"):
        split_dir = cache_root / split
        index = require_file(split_dir / "index.csv", "index.csv")
        split_shards = sorted(split_dir.glob("shard_*.pt"))
        if {path.name for path in split_dir.iterdir()} != {
            "index.csv",
            *(path.name for path in split_shards),
        }:
            raise RuntimeError(f"Unexpected file in finalized {split} cache")
        if audit.get("indexes", {}).get(split) != producer_attestation(index):
            raise RuntimeError(f"Frozen {split} index changed")
        indexes[split] = index
    validate_indexes(torch, cache_root, indexes, accepted)
    shard_config = load_shard_config(torch, shards)
    config_sha = sha256_bytes(json_bytes(config))
    independent = audit_shards(torch, cache_root, accepted_ids, shard_config, config_sha)
    compare_audits(independent, audit)
    return {
        "root": cache_root,
        "config_path": config_path,
        "config": config,
        "audit_path": audit_path,
        "audit": audit,
        "indexes": indexes,
        "shards": shards,
        "shard_config": shard_config,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    provenance = validate_finalized(
        args.production_plan, args.accepted, args.selection, args.qa_report
    )
    accepted = read_jsonl(args.accepted.expanduser().resolve())
    if provenance["rows"] != EXPECTED_ROWS:
        raise RuntimeError("Production plan scope changed")
    if len({row["speaker_id"] for row in accepted}) > EXPECTED_SPEAKERS:
        raise RuntimeError("Accepted speaker scope exceeds the VIVOS source scope")
    for row in accepted:
        source = row["source_provenance"]
        if (
            row.get("eligibility_split") not in {"train", "dev"}
            or source.get("license") != LICENSE
            or source.get("source_repo") != SOURCE_REPO
            or source.get("corpus_revision") != SOURCE_REVISION
            or source.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        ):
            raise RuntimeError(f"Accepted source/license provenance changed: {row.get('id')}")
    cache = validate_cache(args.cache_root, accepted)
    if cache["config"].get("inputs") != provenance or cache["audit"].get("inputs") != provenance:
        raise RuntimeError("Cache audit inputs differ from cache build config")
    return {
        "provenance": provenance,
        "accepted": accepted,
        "cache": cache,
    }


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def build_archive(destination: Path, cache_root: Path, split: str) -> None:
    require_version("zstandard", ZSTANDARD_VERSION)
    import zstandard

    members = [cache_root / split / "index.csv", *sorted((cache_root / split).glob("shard_*.pt"))]
    temporary_tar = destination.with_suffix("")
    if destination.exists() or temporary_tar.exists():
        raise RuntimeError(f"Refusing to overwrite archive: {destination}")
    try:
        with tarfile.open(temporary_tar, "x", format=tarfile.USTAR_FORMAT) as archive:
            for source in members:
                info = normalized_tar_info(
                    archive.gettarinfo(str(source), arcname=f"{split}/{source.name}")
                )
                with source.open("rb") as input_file:
                    archive.addfile(info, input_file)
        compressor = zstandard.ZstdCompressor(
            level=19,
            threads=0,
            write_checksum=True,
            write_content_size=True,
        )
        with temporary_tar.open("rb") as source, destination.open("xb") as output:
            compressor.copy_stream(source, output, size=temporary_tar.stat().st_size)
            output.flush()
            os.fsync(output.fileno())
    finally:
        temporary_tar.unlink(missing_ok=True)


def index_summary(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return len(rows), sum(int(row["source_frames"]) for row in rows)


def dataset_card(cache: dict[str, Any]) -> bytes:
    train_rows, train_frames = index_summary(cache["indexes"]["train"])
    dev_rows, dev_frames = index_summary(cache["indexes"]["dev"])
    audit = cache["audit"]
    exclusions = cache["config"]["inputs"]["speaker_exclusions"]
    excluded = ", ".join(exclusions["speaker_ids"]) or "none"
    text = f"""---
license: cc-by-nc-sa-4.0
language:
- vi
- en
task_categories:
- translation
pretty_name: Hibiki-Zero VIVOS VI-to-EN Mimi Cache v2
---

# Hibiki-Zero VIVOS VI→EN Mimi cache v2

This immutable prefix contains provenance-preserving PyTorch-Mimi cache v2
shards derived from real Vietnamese VIVOS source speech and the validated
Qwen3-TTS MLX retry-v6 English targets. It contains discrete training codes and
metadata, not source or target WAV files.

| Split | Rows | Vietnamese source hours |
|---|---:|---:|
| train | {train_rows:,} | {train_frames / 12.5 / 3600:.6f} |
| dev | {dev_rows:,} | {dev_frames / 12.5 / 3600:.6f} |

Cache schema: `{CACHE_FORMAT}`. Source license: {LICENSE}. The build audit
reports `{audit["cache_rows"]}` rows, zero missing/duplicate/unexpected ids, and
zero invalid or degenerate code rows. `SHA256SUMS` covers every published file;
`release_manifest.json` records the exact build inputs and archive toolchain.
The release also preserves every executed generation/QA attempt and explicitly
records which of retry rounds 1 and 2 were not executed after the terminal GO.
Release-level quality exclusions: `{excluded}` ({exclusions["rows"]} source
rows). Their generated artifacts and QA remain in provenance metadata but are
not present in the training cache.

Extract both archives into one directory. Keep this VIVOS real-source cache
separate from FLEURS and PhoMT so the trainer can enforce source-aware sampling.
"""
    return text.encode()


def attribution() -> bytes:
    return f"""# Attribution

This cache release is derived from the VIVOS Vietnamese speech corpus published
by AILAB-VNUHCM at `{SOURCE_REPO}`, pinned to revision `{SOURCE_REVISION}`.
The verified source archive SHA-256 is `{SOURCE_ARCHIVE_SHA256}`.

VIVOS is licensed under {LICENSE}. Credit the VIVOS authors and corpus when
using or redistributing this derived release. English target speech was generated
with the pinned Qwen3-TTS MLX retry-v6 campaign recorded in
`metadata/campaign/` and `metadata/attempts/`; Mimi codes were produced with the pinned PyTorch backend recorded in
`metadata/cache/cache_config.json`.
""".encode()


def license_text() -> bytes:
    return f"""# License

This VIVOS-derived cache release is distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license ({LICENSE}).

License terms: https://creativecommons.org/licenses/by-nc-sa/4.0/

You must provide attribution, use the material only for non-commercial purposes,
and distribute adaptations under the same license. This notice does not replace
the license terms or the upstream VIVOS attribution requirements.
""".encode()


def release_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != LOCAL_REPORT)


def write_sha256sums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    files = [path for path in release_files(root) if path != checksum_path]
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files]
    immutable_write(checksum_path, "".join(lines).encode())


def parse_sha256sums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    path = root / "SHA256SUMS"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, relative = line.partition("  ")
        relative_path = Path(relative)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in checksums
        ):
            raise RuntimeError(f"Invalid SHA256SUMS line {line_number}")
        checksums[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in release_files(root)
        if path.name != "SHA256SUMS"
    }
    if set(checksums) != actual:
        raise RuntimeError("SHA256SUMS does not exactly cover the release")
    for relative, digest in checksums.items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"Release checksum mismatch: {relative}")
    return checksums


def packaged_sources(values: dict[str, Any]) -> dict[str, Path]:
    provenance = values["provenance"]
    cache = values["cache"]

    def artifact(key: str) -> Path:
        return Path(str(provenance[key]["path"])).resolve()

    sources = {
        PACKAGED_INPUTS["cache_config"]: cache["config_path"],
        PACKAGED_INPUTS["cache_audit"]: cache["audit_path"],
        PACKAGED_INPUTS["train_index"]: cache["indexes"]["train"],
        PACKAGED_INPUTS["dev_index"]: cache["indexes"]["dev"],
        PACKAGED_INPUTS["source_audit_report"]: artifact("source_audit_report"),
        PACKAGED_INPUTS["source_audit_rows"]: artifact("source_audit_rows"),
        PACKAGED_INPUTS["reference_map"]: artifact("reference_map"),
        PACKAGED_INPUTS["reference_report"]: artifact("reference_report"),
        PACKAGED_INPUTS["production_plan"]: artifact("production_plan"),
        PACKAGED_INPUTS["production_attestation"]: artifact("production_attestation"),
        PACKAGED_INPUTS["policy"]: artifact("policy"),
        PACKAGED_INPUTS["validation_go"]: artifact("validation_go"),
        PACKAGED_INPUTS["approval"]: artifact("approval"),
        PACKAGED_INPUTS["source_plan"]: artifact("source_plan"),
        PACKAGED_INPUTS["accepted"]: artifact("accepted"),
        PACKAGED_INPUTS["rejected"]: artifact("rejected"),
        PACKAGED_INPUTS["selection"]: artifact("selection"),
        PACKAGED_INPUTS["selected_candidates"]: artifact("selected_candidates"),
        PACKAGED_INPUTS["qa_report"]: artifact("qa_report"),
        PACKAGED_INPUTS["selection_report"]: artifact("selection_report"),
        PACKAGED_INPUTS["manual_required"]: artifact("manual_required"),
        PACKAGED_INPUTS["manual_evidence"]: Path(
            str(provenance["manual_evidence"]["artifact"]["path"])
        ).resolve(),
    }
    for attempt in provenance["attempts"]:
        if attempt.get("state") == "not_executed_after_terminal_go":
            continue
        name = attempt["attempt_name"]
        for key, filename in (
            ("generation_manifest", "generation_manifest.json"),
            ("candidate", "candidate.json"),
            ("raw_results", "raw_results.jsonl"),
            ("qa_report", "qa_report.json"),
            ("qa_metrics", "metrics.jsonl"),
        ):
            sources[f"metadata/attempts/{name}/{filename}"] = Path(
                str(attempt[key]["path"])
            ).resolve()
        if attempt.get("retry_manifest"):
            sources[f"metadata/attempts/{name}/retry_manifest.jsonl"] = Path(
                str(attempt["retry_manifest"]["path"])
            ).resolve()
        for record in attempt["group_records"]:
            path = Path(str(record["path"])).resolve()
            sources[f"metadata/attempts/{name}/groups/{path.parent.name}.json"] = path
        qa_dir = Path(str(attempt["qa_report"]["path"])).resolve().parent
        for timing in sorted(qa_dir.glob("*timing*.json")):
            sources[f"metadata/attempts/{name}/timings/{timing.name}"] = timing
        for embedding in sorted(qa_dir.glob("reference_embeddings/*.json")):
            sources[f"metadata/attempts/{name}/reference_embeddings/{embedding.name}"] = embedding
    production_root = artifact("production_plan").parent
    for path in sorted(production_root.glob("launch_record*.json")):
        sources[f"metadata/campaign/launch/{path.name}"] = path
    for path in sorted(production_root.glob("generation_attempt*.log")):
        sources[f"metadata/campaign/logs/{path.name}"] = path
    report_root = artifact("validation_go").parent
    for name in ("RNG_ERRATUM_2026-08-04.md", "POSTPROCESS_IMPLEMENTATION_2026-08-04.md"):
        path = report_root / name
        if path.is_file():
            sources[f"metadata/campaign/{name}"] = path
    for pattern in ("*timing*.json", "*timing*.log"):
        for path in sorted(report_root.glob(pattern)):
            sources[f"metadata/validation_timings/{path.name}"] = path
    for root_name, root in (
        ("final", artifact("qa_report").parent),
        ("selection", artifact("selection_report").parent),
    ):
        for pattern in ("command_history.jsonl", "*timing*.json", "*.log"):
            for path in sorted(root.glob(pattern)):
                sources[f"metadata/qa_execution/{root_name}/{path.name}"] = path
    bound = set(provenance_paths(provenance))
    missing = sorted(str(path) for path in bound - set(sources.values()))
    if missing:
        raise RuntimeError(f"Release mapping omitted finalized provenance: {missing[:5]}")
    return sources


def prepare(args: argparse.Namespace) -> None:
    if RELEASE_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite immutable release: {RELEASE_DIR}")
    values = validate_inputs(args)
    RELEASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{RELEASE_ID}.", dir=RELEASE_DIR.parent))
    try:
        cache = values["cache"]
        sources = packaged_sources(values)
        for relative, source in sorted(sources.items()):
            copy_new(source, staging / relative)
        (staging / "archives").mkdir()
        build_archive(staging / "archives/cache_train.tar.zst", cache["root"], "train")
        build_archive(staging / "archives/cache_dev.tar.zst", cache["root"], "dev")
        immutable_write(staging / "README.md", dataset_card(cache))
        immutable_write(staging / "ATTRIBUTION.md", attribution())
        immutable_write(staging / "LICENSE.md", license_text())
        component_records = {
            path.relative_to(staging).as_posix(): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in release_files(staging)
        }
        manifest = {
            "schema_version": SCHEMA,
            "release_id": RELEASE_ID,
            "repository": REPO_ID,
            "remote_prefix": REMOTE_PREFIX,
            "cache_format": CACHE_FORMAT,
            "scope": cache["config"]["scope"],
            "attempts": values["provenance"]["attempts"],
            "manual_evidence": values["provenance"]["manual_evidence"],
            "speaker_exclusions": values["provenance"]["speaker_exclusions"],
            "source": {
                "repo_id": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "archive_sha256": SOURCE_ARCHIVE_SHA256,
                "license": LICENSE,
            },
            "toolchain": {
                "repository_commit": git_commit(),
                "release_script_sha256": sha256_file(Path(__file__).resolve()),
                "torch": TORCH_VERSION,
                "zstandard": ZSTANDARD_VERSION,
            },
            "components": component_records,
        }
        immutable_write(staging / "release_manifest.json", json_bytes(manifest))
        write_sha256sums(staging)
        parse_sha256sums(staging)
        staging.rename(RELEASE_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared immutable release: {RELEASE_DIR}")


def validate_release(root: Path) -> tuple[dict[str, str], list[Path]]:
    if root.resolve() != RELEASE_DIR.resolve() or not root.is_dir():
        raise RuntimeError(f"Expected immutable release directory: {RELEASE_DIR}")
    if (root / LOCAL_REPORT).exists():
        raise RuntimeError("Release already has a successful publication report")
    checksums = parse_sha256sums(root)
    files = release_files(root)
    required = set(PACKAGED_INPUTS.values()) | {
        "archives/cache_train.tar.zst",
        "archives/cache_dev.tar.zst",
        "README.md",
        "ATTRIBUTION.md",
        "LICENSE.md",
        "release_manifest.json",
        "SHA256SUMS",
    }
    actual = {path.relative_to(root).as_posix() for path in files}
    if not required <= actual:
        raise RuntimeError(f"Local release is missing required files: {sorted(required - actual)}")
    for name in ("cache_train.tar.zst", "cache_dev.tar.zst"):
        if (root / "archives" / name).stat().st_size <= MIN_LFS_ARCHIVE_BYTES:
            raise RuntimeError(
                f"Cache archive is too small for mandatory Hub LFS verification: {name}"
            )
    manifest = read_json(root / "release_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("release_id") != RELEASE_ID
        or manifest.get("repository") != REPO_ID
        or manifest.get("remote_prefix") != REMOTE_PREFIX
        or not isinstance(manifest.get("attempts"), list)
        or [attempt.get("attempt") for attempt in manifest["attempts"]] != [0, 1, 2]
        or manifest["attempts"][0].get("state") == "not_executed_after_terminal_go"
    ):
        raise RuntimeError("Release manifest contract changed")
    for attempt in manifest["attempts"]:
        prefix = f"metadata/attempts/{attempt['attempt_name']}/"
        present = {path for path in actual if path.startswith(prefix)}
        if attempt.get("state") == "not_executed_after_terminal_go":
            if present:
                raise RuntimeError(f"Absent attempt was packaged: {attempt['attempt_name']}")
            continue
        required_attempt = {
            f"{prefix}generation_manifest.json",
            f"{prefix}candidate.json",
            f"{prefix}raw_results.jsonl",
            f"{prefix}qa_report.json",
            f"{prefix}metrics.jsonl",
        }
        if not required_attempt <= present or not any(
            path.startswith(f"{prefix}groups/") for path in present
        ):
            raise RuntimeError(
                f"Executed attempt metadata is incomplete: {attempt['attempt_name']}"
            )
    components = manifest.get("components", {})
    component_paths = [
        path for path in files if path.name not in {"release_manifest.json", "SHA256SUMS"}
    ]
    expected_components = {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in component_paths
    }
    if components != expected_components:
        raise RuntimeError("Release manifest component set or hash changed")
    return checksums, files


def load_hf_token(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise RuntimeError(f"Token file is not a regular file: {resolved}")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"Token file must not grant group/other permissions: {oct(mode)}")
    values = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HF_TOKEN":
            token = value.strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
                token = token[1:-1]
            values.append(token)
    if len(values) != 1 or not values[0]:
        raise RuntimeError("Mode-safe .env must contain exactly one non-empty HF_TOKEN")
    return values[0]


def verify_remote_info(info: Any, files: list[Path]) -> list[dict[str, Any]]:
    expected = {
        f"{REMOTE_PREFIX}/{path.relative_to(RELEASE_DIR).as_posix()}": path for path in files
    }
    siblings = {
        sibling.rfilename: sibling
        for sibling in info.siblings
        if sibling.rfilename.startswith(f"{REMOTE_PREFIX}/")
    }
    if set(siblings) != set(expected):
        raise RuntimeError("Remote release prefix does not exactly match the local release")
    records = []
    for remote_path, local_path in sorted(expected.items()):
        sibling = siblings[remote_path]
        size = local_path.stat().st_size
        digest = sha256_file(local_path)
        if sibling.size != size or not sibling.blob_id:
            raise RuntimeError(f"Remote path/size/blob metadata mismatch: {remote_path}")
        if len(sibling.blob_id) != 40 or any(
            character not in "0123456789abcdef" for character in sibling.blob_id
        ):
            raise RuntimeError(f"Remote Git blob id is invalid: {remote_path}")
        lfs_sha = sibling.lfs.sha256 if sibling.lfs is not None else None
        # Hub metadata exposes SHA-256 only for LFS objects. Every LFS file is
        # matched here; the clean snapshot verifies SHA-256 for regular Git
        # metadata files as well. Both cache archives must be LFS.
        if sibling.lfs is not None and (sibling.lfs.size != size or lfs_sha != digest):
            raise RuntimeError(f"Remote LFS SHA-256 mismatch: {remote_path}")
        if local_path.suffix == ".zst" and lfs_sha != digest:
            raise RuntimeError(f"Cache archive is not hash-verified LFS: {remote_path}")
        records.append(
            {
                "path": remote_path,
                "size": size,
                "sha256": digest,
                "blob_id": sibling.blob_id,
                "lfs_sha256": lfs_sha,
            }
        )
    return records


def safe_extract_zstd(archive_path: Path, destination: Path) -> None:
    import zstandard

    destination = destination.resolve()
    with archive_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(destination) or not member.isfile():
                        raise RuntimeError(f"Unsafe archive member: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Unreadable archive member: {member.name}")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def clean_room_verify(download_root: Path) -> dict[str, Any]:
    require_version("torch", TORCH_VERSION)
    import torch

    release = download_root / REMOTE_PREFIX
    checksums = parse_sha256sums(release)
    with tempfile.TemporaryDirectory(prefix="hibiki-vivos-release-extract-") as extract_name:
        extracted = Path(extract_name).resolve()
        safe_extract_zstd(release / "archives/cache_train.tar.zst", extracted)
        safe_extract_zstd(release / "archives/cache_dev.tar.zst", extracted)
        accepted = read_jsonl(release / PACKAGED_INPUTS["accepted"])
        accepted_ids = {str(row.get("id", "")) for row in accepted}
        cache_config_path = release / PACKAGED_INPUTS["cache_config"]
        cache_config = read_json(cache_config_path)
        frozen_audit = read_json(release / PACKAGED_INPUTS["cache_audit"])
        shards = sorted([*extracted.glob("train/shard_*.pt"), *extracted.glob("dev/shard_*.pt")])
        shard_config = load_shard_config(torch, shards)
        independent = audit_shards(
            torch,
            extracted,
            accepted_ids,
            shard_config,
            sha256_bytes(json_bytes(cache_config)),
        )
        compare_audits(independent, frozen_audit)
        frozen_shards = {
            f"{Path(str(record['path'])).parent.name}/{Path(str(record['path'])).name}": record[
                "sha256"
            ]
            for record in frozen_audit["shards"]
        }
        extracted_shards = {
            path.relative_to(extracted).as_posix(): sha256_file(path) for path in shards
        }
        if extracted_shards != frozen_shards:
            raise RuntimeError("Clean-room shard hashes differ from the cache audit")
        for split in ("train", "dev"):
            extracted_index = extracted / split / "index.csv"
            packaged_index = release / PACKAGED_INPUTS[f"{split}_index"]
            if sha256_file(extracted_index) != sha256_file(packaged_index):
                raise RuntimeError(f"Clean-room {split} index changed")
    return {
        "sha256sums_verified": len(checksums),
        "cache_rows": independent["cache_rows"],
        "cache_audit_complete": independent["complete"],
        "invalid_rows": len(independent["invalid_rows"]),
    }


def publish(args: argparse.Namespace) -> None:
    require_version("huggingface-hub", HF_HUB_VERSION)
    require_version("zstandard", ZSTANDARD_VERSION)
    require_version("torch", TORCH_VERSION)
    checksums, files = validate_release(RELEASE_DIR)
    token = load_hf_token(args.env_file)
    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{REMOTE_PREFIX}/{path.relative_to(RELEASE_DIR).as_posix()}",
            path_or_fileobj=path,
        )
        for path in files
    ]
    head_info = api.dataset_info(REPO_ID, revision="main", files_metadata=True, token=token)
    if any(
        sibling.rfilename == REMOTE_PREFIX or sibling.rfilename.startswith(f"{REMOTE_PREFIX}/")
        for sibling in head_info.siblings
    ):
        raise RuntimeError(f"Remote destination prefix is not empty: {REMOTE_PREFIX}")
    head = str(head_info.sha)
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("Dataset HEAD is not a full Git commit OID")
    commit = api.create_commit(
        REPO_ID,
        operations,
        commit_message=f"Publish VIVOS Mimi cache v2: {RELEASE_ID}",
        repo_type="dataset",
        revision="main",
        parent_commit=head,
        token=token,
    )
    commit_oid = str(commit.oid)
    if len(commit_oid) != 40 or any(
        character not in "0123456789abcdef" for character in commit_oid
    ):
        raise RuntimeError("Hub returned an invalid commit OID")
    committed = api.dataset_info(REPO_ID, revision=commit_oid, files_metadata=True, token=token)
    if str(committed.sha) != commit_oid:
        raise RuntimeError("Returned commit OID did not resolve exactly")
    remote_files = verify_remote_info(committed, files)
    with tempfile.TemporaryDirectory(prefix="hibiki-vivos-release-download-") as download_name:
        download_root = Path(download_name).resolve()
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=commit_oid,
            allow_patterns=[
                f"{REMOTE_PREFIX}/{path.relative_to(RELEASE_DIR).as_posix()}" for path in files
            ],
            local_dir=download_root,
            token=token,
        )
        clean_room = clean_room_verify(download_root)
    report = {
        "schema_version": SCHEMA,
        "release_id": RELEASE_ID,
        "repository": REPO_ID,
        "remote_prefix": REMOTE_PREFIX,
        "parent_commit": head,
        "commit_oid": commit_oid,
        "local_sha256sums": len(checksums),
        "remote_files": remote_files,
        "clean_room": clean_room,
        "huggingface_hub": HF_HUB_VERSION,
    }
    immutable_write(RELEASE_DIR / LOCAL_REPORT, json_bytes(report))
    print(f"Published and clean-room verified {REPO_ID}@{commit_oid}")


def main() -> None:
    args = parse_args()
    if args.action == "preflight-historical":
        print(
            json.dumps(
                validate_historical(args.policy, args.qa_root, args.selection_report),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.action == "preflight":
        final_paths = (args.accepted, args.selection, args.qa_report)
        if any(final_paths) and not all(final_paths):
            raise RuntimeError(
                "Preflight requires all or none of --accepted, --selection, and --qa-report"
            )
        try:
            result = (
                validate_finalized(
                    args.production_plan, args.accepted, args.selection, args.qa_report
                )
                if all(final_paths)
                else validate_live_state(args.production_plan)
            )
            if result.get("state") == "ready":
                if args.cache_root is None:
                    raise IncompleteCampaign("Final QA is ready but --cache-root was not supplied")
                accepted = read_jsonl(args.accepted.expanduser().resolve())
                cache = validate_cache(args.cache_root, accepted)
                result["cache"] = {
                    "root": str(cache["root"]),
                    "rows": cache["audit"]["cache_rows"],
                    "complete": cache["audit"]["complete"],
                }
                result["release_ready"] = True
        except IncompleteCampaign as error:
            result = {"state": "incomplete", "release_ready": False, "reason": str(error)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("release_ready") is not True:
            raise SystemExit(3)
        return
    prepare(args) if args.action == "prepare" else publish(args)


if __name__ == "__main__":
    main()

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

SCHEMA = "hibiki_vivos_cache_release_v1"
RELEASE_ID = "vivos_qwen3_tts_mlx_v3_full_v1"
RELEASE_DIR = REPO_ROOT / "releases" / "v2" / RELEASE_ID
REMOTE_PREFIX = f"v2/{RELEASE_ID}"
REPO_ID = "huybik/hibiki-zero-vi-full-sft"
HF_HUB_VERSION = "1.21.0"
ZSTANDARD_VERSION = "0.23.0"
TORCH_VERSION = "2.13.0"
QA_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_qa_v1"
CAMPAIGN_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_v1"
SOURCE_AUDIT_SCHEMA = "hibiki_vivos_source_asr_mps_full_v1"
REFERENCE_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_v3_reference_map_v1"
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
    "campaign_config": "metadata/campaign/campaign_config.json",
    "approval": "metadata/campaign/approval_override.json",
    "plan": "metadata/campaign/generation_plan.jsonl",
    "attempts": "metadata/campaign/generation_attempts.jsonl",
    "accepted": "metadata/qa/accepted.jsonl",
    "rejected": "metadata/qa/rejected.jsonl",
    "selection": "metadata/qa/selection.jsonl",
    "qa_report": "metadata/qa/aggregate_report.json",
    "manual_required": "metadata/qa/manual_review_required.tsv",
    "manual_review": "metadata/qa/manual_review.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--source-audit-report", type=Path, required=True)
    prepare.add_argument("--source-audit-rows", type=Path, required=True)
    prepare.add_argument("--reference-map", type=Path, required=True)
    prepare.add_argument("--reference-report", type=Path, required=True)
    prepare.add_argument("--campaign-config", type=Path, required=True)
    prepare.add_argument("--approval", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--attempts", type=Path, required=True)
    prepare.add_argument("--accepted", type=Path, required=True)
    prepare.add_argument("--rejected", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--qa-report", type=Path, required=True)
    prepare.add_argument("--manual-required", type=Path, required=True)
    prepare.add_argument("--manual-review", type=Path, required=True)
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


def require_attestation(path: Path, expected: dict[str, Any], label: str) -> None:
    actual = producer_attestation(path)
    if actual != {
        "path": str(Path(str(expected.get("path", ""))).resolve()),
        "sha256": expected.get("sha256"),
    }:
        raise RuntimeError(f"{label} is not the frozen producer artifact: {path}")


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


def validate_manual_review(required_path: Path, review_path: Path, report: dict[str, Any]) -> None:
    required_record = report.get("manual_review", {})
    if required_path.resolve() != Path(str(required_record.get("required_tsv", ""))).resolve():
        raise RuntimeError("Manual requirement TSV is not bound to the QA report")
    if producer_attestation(review_path) != required_record.get("review_file"):
        raise RuntimeError("Completed manual review is not bound to the QA report")
    with required_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        columns = {"candidate_id", "status", "prompt_leak", "notes"}
        if not reader.fieldnames or not columns.issubset(reader.fieldnames):
            raise RuntimeError("Manual requirement TSV contract changed")
        required_rows = list(reader)
    required_ids = [str(row["candidate_id"]).strip() for row in required_rows]
    expected_required = set(required_record.get("seeded_sample_candidates", [])) | set(
        required_record.get("failed_candidates", [])
    )
    if (
        len(required_ids) != len(set(required_ids))
        or set(required_ids) != expected_required
        or len(required_ids) != int(required_record.get("required_candidates", -1))
    ):
        raise RuntimeError("Manual requirement candidate set changed")
    with review_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or not columns.issubset(reader.fieldnames):
            raise RuntimeError("Manual review TSV contract changed")
        reviewed = {}
        for row in reader:
            candidate_id = str(row["candidate_id"]).strip()
            status = str(row["status"]).strip().casefold()
            prompt_leak = str(row["prompt_leak"]).strip().casefold()
            if not status and not prompt_leak:
                continue
            if (
                not candidate_id
                or candidate_id in reviewed
                or status not in {"pass", "fail"}
                or prompt_leak not in {"yes", "no"}
            ):
                raise RuntimeError("Completed manual review contains an invalid row")
            reviewed[candidate_id] = row
    if (
        set(required_record.get("missing_candidates", []))
        or len(reviewed) < int(required_record.get("required_candidates", -1))
        or not expected_required.issubset(reviewed)
    ):
        raise RuntimeError("Manual review is incomplete")


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "source_audit_report": require_file(args.source_audit_report, "audit_report.json"),
        "source_audit_rows": require_file(args.source_audit_rows, "row_metrics.jsonl"),
        "reference_map": require_file(args.reference_map, "reference_map.jsonl"),
        "reference_report": require_file(args.reference_report, "reference_map_report.json"),
        "campaign_config": require_file(args.campaign_config, "campaign_config.json"),
        "approval": require_file(args.approval, "approval_override.json"),
        "plan": require_file(args.plan, "generation_plan.jsonl"),
        "attempts": require_file(args.attempts, "generation_attempts.jsonl"),
        "accepted": require_file(args.accepted, "accepted.jsonl"),
        "rejected": require_file(args.rejected, "rejected.jsonl"),
        "selection": require_file(args.selection, "selection.jsonl"),
        "qa_report": require_file(args.qa_report, "aggregate_report.json"),
        "manual_required": require_file(args.manual_required, "manual_review_required.tsv"),
        "manual_review": args.manual_review.expanduser().resolve(),
    }
    if not paths["manual_review"].is_file():
        raise RuntimeError(f"Completed manual review is missing: {paths['manual_review']}")

    report = read_json(paths["qa_report"])
    if (
        report.get("schema_version") != QA_SCHEMA
        or report.get("decision") != "go"
        or not all(report.get("machine_checks", {}).values())
        or not all(report.get("manual_checks", {}).values())
    ):
        raise RuntimeError("Final QA report has not passed all machine and manual gates")
    outputs = report.get("outputs", {})
    for key in ("accepted", "rejected", "selection"):
        require_attestation(paths[key], outputs.get(key, {}), f"QA {key}")
    qa_inputs = report.get("inputs", {})
    require_attestation(paths["plan"], qa_inputs.get("plan", {}), "Generation plan")
    require_attestation(
        paths["attempts"], qa_inputs.get("generation_manifest", {}), "Generation attempts"
    )
    campaign_artifacts = qa_inputs.get("campaign_artifacts", {})
    artifact_keys = {
        "campaign_config": "campaign_config",
        "approval": "approval",
        "source_audit_report": "source_audit_report",
        "source_audit_rows": "source_audit_rows",
        "reference_map": "reference_map",
        "reference_report": "reference_report",
    }
    for path_key, report_key in artifact_keys.items():
        require_attestation(paths[path_key], campaign_artifacts.get(report_key, {}), path_key)
    validate_manual_review(paths["manual_required"], paths["manual_review"], report)

    campaign = read_json(paths["campaign_config"])
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise RuntimeError("Campaign config schema changed")
    require_attestation(paths["plan"], campaign.get("plan", {}), "Campaign plan")
    require_attestation(paths["approval"], campaign.get("approval", {}), "Campaign approval")
    for key, path_key in (("report", "source_audit_report"), ("row_metrics", "source_audit_rows")):
        require_attestation(
            paths[path_key], campaign.get("source_audit", {}).get(key, {}), path_key
        )
    for key, path_key in (("map", "reference_map"), ("report", "reference_report")):
        require_attestation(paths[path_key], campaign.get("references", {}).get(key, {}), path_key)

    source_report = read_json(paths["source_audit_report"])
    if (
        source_report.get("schema_version") != SOURCE_AUDIT_SCHEMA
        or source_report.get("complete") is not True
        or source_report.get("rows") != EXPECTED_ROWS
        or source_report.get("speakers") != EXPECTED_SPEAKERS
    ):
        raise RuntimeError("Full source audit is incomplete")
    require_attestation(
        paths["source_audit_rows"], source_report.get("row_metrics", {}), "Source audit rows"
    )
    references = read_jsonl(paths["reference_map"])
    reference_report = read_json(paths["reference_report"])
    if (
        any(row.get("schema_version") != REFERENCE_SCHEMA for row in references)
        or len(references) != EXPECTED_SPEAKERS
        or reference_report.get("schema_version") != REFERENCE_SCHEMA
        or reference_report.get("status") != "complete"
        or reference_report.get("references") != references
    ):
        raise RuntimeError("Reference map/report contract changed")

    plan = read_jsonl(paths["plan"])
    attempts = read_jsonl(paths["attempts"])
    accepted = read_jsonl(paths["accepted"])
    rejected = read_jsonl(paths["rejected"])
    selection = read_jsonl(paths["selection"])
    plan_ids = [str(row.get("id", "")) for row in plan]
    plan_id_set = set(plan_ids)
    accepted_ids = {str(row.get("id", "")) for row in accepted}
    rejected_ids = {str(row.get("id", "")) for row in rejected}
    selection_ids = [str(row.get("id", "")) for row in selection]
    selection_by_id = {str(row.get("id", "")): row for row in selection}
    attempt_keys = {(str(row.get("id", "")), int(row.get("attempt", -1))) for row in attempts}
    if (
        not plan
        or len(plan) != EXPECTED_ROWS
        or any(row.get("schema_version") != CAMPAIGN_SCHEMA for row in plan)
        or len(plan_ids) != len(set(plan_ids))
        or selection_ids != plan_ids
        or len(accepted_ids) != len(accepted)
        or len(rejected_ids) != len(rejected)
        or accepted_ids & rejected_ids
        or accepted_ids | rejected_ids != plan_id_set
        or len(attempt_keys) != len(attempts)
        or any(
            row_id not in plan_id_set or attempt not in {0, 1} for row_id, attempt in attempt_keys
        )
        or any(selection_by_id[row_id].get("status") != "accepted" for row_id in accepted_ids)
        or any(selection_by_id[row_id].get("status") != "rejected" for row_id in rejected_ids)
        or any(row.get("eligibility_split") not in {"train", "dev"} for row in accepted)
        or any(row.get("source_provenance", {}).get("license") != LICENSE for row in accepted)
        or any(
            row.get("source_provenance", {}).get("source_repo") != SOURCE_REPO for row in accepted
        )
        or any(
            row.get("source_provenance", {}).get("corpus_revision") != SOURCE_REVISION
            for row in accepted
        )
        or any(
            row.get("source_provenance", {}).get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
            for row in accepted
        )
        or campaign.get("scope", {}).get("rows") != EXPECTED_ROWS
        or campaign.get("scope", {}).get("speakers") != EXPECTED_SPEAKERS
        or campaign.get("scope", {}).get("test_sealed") is not True
        or report.get("scope", {}).get("plan_rows") != EXPECTED_ROWS
    ):
        raise RuntimeError("Plan/attempt/selection/license scope changed")

    cache = validate_cache(args.cache_root, accepted)
    cache_inputs = cache["config"].get("inputs", {})
    for key in ("plan", "accepted", "selection", "qa_report", "campaign_config"):
        require_attestation(paths[key], cache_inputs.get(key, {}), f"Cache input {key}")
    if cache["audit"].get("inputs") != cache_inputs:
        raise RuntimeError("Cache audit inputs differ from cache build config")
    return {
        "paths": paths,
        "report": report,
        "plan": plan,
        "accepted": accepted,
        "accepted_ids": accepted_ids,
        "rejected": rejected,
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
shards derived from real Vietnamese VIVOS source speech and Qwen3-TTS English
targets. It contains discrete training codes and metadata, not source or target
WAV files.

| Split | Rows | Vietnamese source hours |
|---|---:|---:|
| train | {train_rows:,} | {train_frames / 12.5 / 3600:.6f} |
| dev | {dev_rows:,} | {dev_frames / 12.5 / 3600:.6f} |

Cache schema: `{CACHE_FORMAT}`. Source license: {LICENSE}. The build audit
reports `{audit["cache_rows"]}` rows, zero missing/duplicate/unexpected ids, and
zero invalid or degenerate code rows. `SHA256SUMS` covers every published file;
`release_manifest.json` records the exact build inputs and archive toolchain.

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
with the pinned Qwen3-TTS MLX campaign recorded in `metadata/campaign/`; Mimi
codes were produced with the pinned PyTorch backend recorded in
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


def prepare(args: argparse.Namespace) -> None:
    if RELEASE_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite immutable release: {RELEASE_DIR}")
    values = validate_inputs(args)
    RELEASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{RELEASE_ID}.", dir=RELEASE_DIR.parent))
    try:
        paths = values["paths"]
        cache = values["cache"]
        sources = {
            **{key: paths[key] for key in paths},
            "cache_config": cache["config_path"],
            "cache_audit": cache["audit_path"],
            "train_index": cache["indexes"]["train"],
            "dev_index": cache["indexes"]["dev"],
        }
        for key, relative in PACKAGED_INPUTS.items():
            copy_new(sources[key], staging / relative)
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
    expected = set(PACKAGED_INPUTS.values()) | {
        "archives/cache_train.tar.zst",
        "archives/cache_dev.tar.zst",
        "README.md",
        "ATTRIBUTION.md",
        "LICENSE.md",
        "release_manifest.json",
        "SHA256SUMS",
    }
    if {path.relative_to(root).as_posix() for path in files} != expected:
        raise RuntimeError("Local release required-file set changed")
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
    ):
        raise RuntimeError("Release manifest contract changed")
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
    prepare(args) if args.action == "prepare" else publish(args)


if __name__ == "__main__":
    main()

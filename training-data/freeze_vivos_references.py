"""Freeze one deterministic, audited VIVOS clone reference per speaker."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qa_vivos_source import (
    FULL_EXPECTED_MANIFESTS,
    FULL_EXPECTED_ROWS,
    FULL_EXPECTED_SPEAKERS,
    FULL_SCHEMA,
    SOURCE_REVIEW_WER_THRESHOLD,
    SOURCE_ASR_CONFIG,
    SOURCE_WAVEFORM_THRESHOLDS,
    source_hard_gate,
)
from synthesize_vivos import (
    canonical_json,
    immutable_write,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_v3_reference_map_v1"
OUT_DIR_NAME = "vivos_qwen3_tts_mlx_v3_full_v1"
REFERENCE_DURATION_MIN_S = 3.0
REFERENCE_DURATION_MAX_S = 8.0
REFERENCE_WER_MAX = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_report", type=Path)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def immutable_write_json(path: Path, value: object) -> None:
    immutable_write(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def load_source_reviews(
    path: Path | None, required_ids: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, str] | None]:
    if path is None:
        return {}, None
    resolved = path.expanduser().resolve()
    reviews: dict[str, dict[str, str]] = {}
    with resolved.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required_columns = {"id", "status", "notes"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Source review TSV must contain {sorted(required_columns)}: {resolved}"
            )
        for line_number, row in enumerate(reader, start=2):
            row_id = row["id"].strip()
            status = row["status"].strip().casefold()
            if not row_id or row_id in reviews:
                raise RuntimeError(
                    f"Empty or duplicate review id at {resolved}:{line_number}"
                )
            if status not in {"pass", "fail"}:
                raise RuntimeError(
                    f"Invalid source review status at {resolved}:{line_number}"
                )
            reviews[row_id] = {"status": status, "notes": row["notes"].strip()}
    unknown = set(reviews) - required_ids
    unresolved = required_ids - set(reviews)
    if unknown or unresolved:
        raise RuntimeError(
            f"Source review coverage mismatch: unknown={sorted(unknown)}, "
            f"unresolved={sorted(unresolved)}"
        )
    return reviews, {"path": str(resolved), "sha256": sha256_file(resolved)}


def load_full_audit(
    report_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    resolved_report = report_path.expanduser().resolve()
    report = json.loads(resolved_report.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != FULL_SCHEMA
        or report.get("complete") is not True
        or report.get("rows") != FULL_EXPECTED_ROWS
        or report.get("speakers") != FULL_EXPECTED_SPEAKERS
        or report.get("hard_gate_thresholds") != SOURCE_WAVEFORM_THRESHOLDS
        or report.get("asr_config") != SOURCE_ASR_CONFIG
        or report.get("source_review_policy", {}).get("wer_threshold_exclusive")
        != SOURCE_REVIEW_WER_THRESHOLD
        or {row.get("sha256") for row in report.get("source_manifests", [])}
        != set(FULL_EXPECTED_MANIFESTS)
    ):
        raise RuntimeError(f"Full source-audit contract mismatch: {resolved_report}")
    scope = report.get("audit_scope", {})
    if (
        scope.get("name") != "accepted_vivos_train_dev_full_v1"
        or scope.get("expected_rows") != FULL_EXPECTED_ROWS
        or scope.get("expected_speakers") != FULL_EXPECTED_SPEAKERS
        or scope.get("split_contained_speakers") is not True
    ):
        raise RuntimeError(f"Full source-audit scope mismatch: {resolved_report}")
    row_metrics = report.get("row_metrics", {})
    metrics_path = Path(str(row_metrics.get("path", "")))
    if not metrics_path.is_file() or sha256_file(metrics_path) != row_metrics.get(
        "sha256"
    ):
        raise RuntimeError(f"Full source-audit rows mismatch: {resolved_report}")
    rows = read_jsonl(metrics_path)
    ids = [str(row.get("id", "")) for row in rows]
    if (
        len(rows) != FULL_EXPECTED_ROWS
        or len(ids) != len(set(ids))
        or any(not row_id for row_id in ids)
        or len({str(row.get("speaker_id", "")) for row in rows})
        != FULL_EXPECTED_SPEAKERS
    ):
        raise RuntimeError("Full source-audit row coverage is incomplete")
    for row in rows:
        provenance = row.get("source_provenance")
        if (
            row.get("schema_version") != FULL_SCHEMA
            or row.get("asr_config") != SOURCE_ASR_CONFIG
            or row.get("hard_gate_thresholds") != SOURCE_WAVEFORM_THRESHOLDS
            or not isinstance(row.get("hard_failure_reasons"), list)
            or row.get("hard_failure_reasons") != source_hard_gate(row)
            or row.get("hard_gate_pass") != (not row["hard_failure_reasons"])
            or row.get("source_review_required")
            != (float(row["asr_wer"]) > SOURCE_REVIEW_WER_THRESHOLD)
            or not isinstance(provenance, dict)
            or row.get("source_provenance_sha256")
            != sha256_bytes(canonical_json(provenance).encode("utf-8"))
            or row.get("reference_text_vi_sha256")
            != sha256_bytes(str(row.get("reference_text_vi", "")).encode("utf-8"))
            or row.get("models") != report.get("models")
            or row.get("runtime") != report.get("runtime")
        ):
            raise RuntimeError(f"Source-audit row provenance mismatch: {row.get('id')}")
    flagged = {str(row["id"]) for row in rows if row["source_review_required"]}
    if flagged != set(report["source_review_policy"]["flagged_ids"]):
        raise RuntimeError("Source review flags disagree with the full audit report")
    hard_failed = {str(row["id"]) for row in rows if row["hard_failure_reasons"]}
    if hard_failed != set(report["hard_gate_failed_ids"]):
        raise RuntimeError("Hard-gate failures disagree with the full audit report")
    return report, rows, resolved_report


def freeze(args: argparse.Namespace) -> None:
    report, rows, audit_report_path = load_full_audit(args.audit_report)
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.name != OUT_DIR_NAME:
        raise RuntimeError(
            f"Reference output directory must end in {OUT_DIR_NAME}: {out_dir}"
        )
    flagged_ids = {str(row["id"]) for row in rows if row["source_review_required"]}
    reviews, review_attestation = load_source_reviews(args.source_review, flagged_ids)

    rows_by_speaker: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[str(row["speaker_id"])].append(row)
    selected: list[dict[str, Any]] = []
    eligibility_counts: dict[str, int] = {}
    for speaker_id in sorted(rows_by_speaker):
        speaker_rows = rows_by_speaker[speaker_id]
        splits = {str(row["eligibility_split"]) for row in speaker_rows}
        if len(splits) != 1:
            raise RuntimeError(f"Speaker is not split-contained: {speaker_id} {splits}")
        eligible = [
            row
            for row in speaker_rows
            if row["hard_gate_pass"]
            and REFERENCE_DURATION_MIN_S
            <= float(row["duration_s"])
            <= REFERENCE_DURATION_MAX_S
            and float(row["asr_wer"]) <= REFERENCE_WER_MAX
            and (
                not row["source_review_required"]
                or reviews.get(str(row["id"]), {}).get("status") == "pass"
            )
        ]
        eligibility_counts[speaker_id] = len(eligible)
        if not eligible:
            raise RuntimeError(f"No eligible clone reference for speaker {speaker_id}")
        chosen = min(
            eligible,
            key=lambda row: (
                float(row["asr_wer"]),
                float(row["asr_cer"]),
                abs(float(row["duration_s"]) - 4.0),
                str(row["id"]),
            ),
        )
        provenance = chosen["source_provenance"]
        selected.append(
            {
                "schema_version": SCHEMA,
                "speaker_id": speaker_id,
                "eligibility_split": chosen["eligibility_split"],
                "reference_id": chosen["id"],
                "reference_audio_path": chosen["audio_path"],
                "reference_audio_sha256": chosen["audio_sha256"],
                "reference_text_vi": chosen["reference_text_vi"],
                "reference_text_vi_sha256": chosen["reference_text_vi_sha256"],
                "duration_s": chosen["duration_s"],
                "asr_wer": chosen["asr_wer"],
                "asr_cer": chosen["asr_cer"],
                "source_audit_row_sha256": sha256_bytes(
                    canonical_json(chosen).encode("utf-8")
                ),
                "source_audit_report_sha256": sha256_file(audit_report_path),
                "source_manifest": chosen["source_manifest"],
                "source_manifest_sha256": chosen["source_manifest_sha256"],
                "corpus": provenance["corpus"],
                "corpus_revision": provenance["corpus_revision"],
                "source_repo": provenance["source_repo"],
                "source_file": provenance["source_file"],
                "source_archive_sha256": provenance["source_archive_sha256"],
                "license": provenance["license"],
            }
        )
    if len(selected) != FULL_EXPECTED_SPEAKERS:
        raise RuntimeError(f"Expected 46 frozen references, found {len(selected)}")

    map_path = out_dir / "reference_map.jsonl"
    map_bytes = "".join(canonical_json(row) + "\n" for row in selected).encode("utf-8")
    immutable_write(map_path, map_bytes)
    report_path = out_dir / "reference_map_report.json"
    split_counts: defaultdict[str, int] = defaultdict(int)
    for row in selected:
        split_counts[str(row["eligibility_split"])] += 1
    reference_report = {
        "schema_version": SCHEMA,
        "status": "complete",
        "reference_count": len(selected),
        "references": selected,
        "speakers": [row["speaker_id"] for row in selected],
        "split_counts": dict(sorted(split_counts.items())),
        "selection_policy": {
            "duration_s_inclusive": [
                REFERENCE_DURATION_MIN_S,
                REFERENCE_DURATION_MAX_S,
            ],
            "waveform_hard_gate_required": True,
            "asr_wer_max_inclusive": REFERENCE_WER_MAX,
            "source_review_required_resolved_for_references": True,
            "stable_order": ["asr_wer", "asr_cer", "abs(duration_s-4.0)", "id"],
        },
        "eligibility_counts_by_speaker": eligibility_counts,
        "source_audit": {
            "report_path": str(audit_report_path),
            "report_sha256": sha256_file(audit_report_path),
            "row_metrics": report["row_metrics"],
            "schema_version": FULL_SCHEMA,
        },
        "source_review": {
            "attestation": review_attestation,
            "flagged_rows": len(flagged_ids),
            "passed_rows": sum(row["status"] == "pass" for row in reviews.values()),
            "failed_rows": sum(row["status"] == "fail" for row in reviews.values()),
            "unresolved_rows": len(flagged_ids - set(reviews)),
        },
        "reference_map": {"path": str(map_path), "sha256": sha256_bytes(map_bytes)},
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    immutable_write_json(report_path, reference_report)
    print(f"Frozen {len(selected)} speaker references: {map_path}")
    print(f"Reference report: {report_path}")


if __name__ == "__main__":
    freeze(parse_args())

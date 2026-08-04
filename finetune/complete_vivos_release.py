#!/usr/bin/env python
"""Continue a machine-GO VIVOS v6 campaign through cache publication."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finetune.cache_vivos_full import sha256_file


BASE_PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/bin/python")
WAIVER_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_manual_waiver_v6"
MACHINE_CHECKS = {
    "selected_corpus_wer",
    "selected_speaker_cosine_median",
    "zero_selected_prompt_leaks",
    "all_selected_candidates_pass_every_row_gate",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def attest(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_write(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


class Completion:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = Path(__file__).resolve().parent.parent
        self.plan = args.production_plan.expanduser().resolve()
        self.qa_root = args.qa_root.expanduser().resolve()
        self.supervisor_work = args.supervisor_work.expanduser().resolve()
        self.work = args.work_dir.expanduser().resolve()
        self.cache_root = args.cache_root.expanduser().resolve()
        self.dataset_root = args.dataset_root.expanduser().resolve()
        self.gender_files = [path.expanduser().resolve() for path in args.gender_files]
        self.env_file = args.env_file.expanduser().resolve()
        self.poll_seconds = args.poll_seconds
        self.final_root = self.qa_root / "final"
        self.waiver = self.final_root / "manual_review_waiver_unattended_repair1.json"
        self.release_dir = (
            self.dataset_root
            / "releases/v2/vivos_qwen3_tts_mlx_retry_v6_full"
        )
        self.state_path = self.work / "state.json"
        self.events_path = self.work / "events.jsonl"
        self.history_path = self.work / "command_history.jsonl"
        self.config_path = self.work / "config.json"
        self.work.mkdir(parents=True, exist_ok=True)
        self.lock = (self.work / "completion.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another VIVOS release completion worker holds the lock") from error
        self.config = self.load_or_create_config()

    def repository_commit(self) -> str:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def scripts(self) -> list[Path]:
        return [
            Path(__file__).resolve(),
            self.repo / "training-data/qa_vivos_qwen_production_v6.py",
            self.repo / "training-data/validate_vivos_qwen_production_v6.py",
            self.repo / "finetune/cache_vivos_full.py",
            self.repo / "finetune/vivos_v6_provenance.py",
            self.repo / "finetune/release_vivos_cache.py",
        ]

    def stable_config(self) -> dict[str, Any]:
        supervisor_config = self.supervisor_work / "config.json"
        expected_dataset_root = Path("/Volumes/data/datasets/hibiki_vi_v2")
        if self.plan.name != "production_plan_repair1.json":
            raise RuntimeError("Completion requires the allocator-cache repair1 plan")
        if self.dataset_root != expected_dataset_root:
            raise RuntimeError(f"Completion requires dataset root {expected_dataset_root}")
        if self.cache_root != (
            expected_dataset_root / "cache/vivos_qwen3_tts_mlx_retry_v6_mimi_v2"
        ):
            raise RuntimeError("Completion requires the fixed VIVOS Mimi v2 cache root")
        if not self.env_file.is_file() or stat.S_IMODE(self.env_file.stat().st_mode) & 0o077:
            raise RuntimeError("The Hugging Face environment file must exist with mode 600")
        if any(not path.is_file() for path in self.gender_files):
            raise RuntimeError("A VIVOS gender file is missing")
        supervisor = read_json(supervisor_config)
        if supervisor.get("bindings", {}).get("production_plan") != attest(self.plan):
            raise RuntimeError("Supervisor is not bound to the repaired production plan")
        speaker_exclusions = supervisor.get("speaker_exclusions", [])
        return {
            "schema_version": "hibiki_vivos_qwen3_tts_unattended_release_v1",
            "authorization": {
                "source": "user request in active Codex session",
                "instruction": "queue the necessary steps for a complete uploaded dataset",
                "scope": "machine-GO-only manual-listening waiver, Mimi cache build, immutable release, Hub upload, and clean-room verification",
                "speaker_exclusions": speaker_exclusions,
                "machine_no_go_behavior": "halt_without_cache_or_upload",
            },
            "repository_commit": self.repository_commit(),
            "production_plan": attest(self.plan),
            "supervisor_config": attest(supervisor_config),
            "scripts": [attest(path) for path in self.scripts()],
            "qa_root": str(self.qa_root),
            "final_root": str(self.final_root),
            "cache_root": str(self.cache_root),
            "dataset_root": str(self.dataset_root),
            "release_dir": str(self.release_dir),
            "gender_files": [attest(path) for path in self.gender_files],
            "env_file": {
                "path": str(self.env_file),
                "mode": oct(stat.S_IMODE(self.env_file.stat().st_mode)),
                "required_key": "HF_TOKEN",
                "content_not_recorded": True,
            },
            "poll_seconds": self.poll_seconds,
        }

    def load_or_create_config(self) -> dict[str, Any]:
        stable = self.stable_config()
        if self.config_path.is_file():
            existing = read_json(self.config_path)
            if {key: existing.get(key) for key in stable} != stable:
                raise RuntimeError("Existing unattended release config differs")
            return existing
        config = {**stable, "created_utc": utc_now()}
        immutable_write(self.config_path, json_bytes(config))
        return config

    def assert_frozen(self) -> None:
        if self.repository_commit() != self.config["repository_commit"]:
            raise RuntimeError("Repository commit changed after queue creation")
        if [attest(path) for path in self.scripts()] != self.config["scripts"]:
            raise RuntimeError("A bound completion script changed after queue creation")
        if attest(self.plan) != self.config["production_plan"]:
            raise RuntimeError("The repaired production plan changed")
        if attest(self.supervisor_work / "config.json") != self.config["supervisor_config"]:
            raise RuntimeError("The bound supervisor config changed")

    def state(self, name: str, reason: str, **details: Any) -> None:
        row = {
            "schema_version": "hibiki_vivos_qwen3_tts_unattended_release_event_v1",
            "utc": utc_now(),
            "state": name,
            "reason": reason,
            "details": details,
        }
        append(self.events_path, row)
        atomic_write(
            self.state_path,
            json_bytes(
                {
                    **row,
                    "config": attest(self.config_path),
                    "events": str(self.events_path),
                    "command_history": str(self.history_path),
                }
            ),
        )

    def halt(self, reason: str, **details: Any) -> None:
        self.state("halted", reason, **details)
        raise RuntimeError(reason)

    def run_command(self, label: str, command: list[str]) -> None:
        self.assert_frozen()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
        log = self.work / "logs" / f"{timestamp}_{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = {
            "schema_version": "hibiki_vivos_qwen3_tts_unattended_release_command_v1",
            "event": "started",
            "label": label,
            "command": command,
            "cwd": str(self.repo),
            "started_utc": utc_now(),
            "log_path": str(log),
        }
        append(self.history_path, started)
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        started_monotonic = time.monotonic()
        with os.fdopen(descriptor, "wb") as stream:
            result = subprocess.run(command, cwd=self.repo, stdout=stream, stderr=subprocess.STDOUT)
        completed = {
            **started,
            "event": "completed",
            "completed_utc": utc_now(),
            "wall_seconds": time.monotonic() - started_monotonic,
            "returncode": result.returncode,
            "terminal_success": result.returncode == 0,
            "log": attest(log),
        }
        append(self.history_path, completed)
        if result.returncode:
            self.halt(
                "Queued command failed",
                label=label,
                returncode=result.returncode,
                log=completed["log"],
            )

    def wait_for_machine_result(self) -> None:
        observed = None
        while True:
            self.assert_frozen()
            supervisor_state = read_json(self.supervisor_work / "state.json")
            name = str(supervisor_state.get("state", ""))
            if name != observed:
                self.state(
                    "waiting_machine_result",
                    "observed postprocess supervisor state",
                    supervisor_state=name,
                    supervisor_state_record=attest(self.supervisor_work / "state.json"),
                )
                observed = name
            if name == "pending_manual_review":
                return
            if name in {"no_go", "halted"}:
                self.halt(
                    "Machine pipeline did not reach the manual-review boundary",
                    supervisor_state=name,
                )
            time.sleep(self.poll_seconds)

    def create_waiver(self) -> Path:
        aggregate_path = self.final_root / "aggregate_report.json"
        aggregate = read_json(aggregate_path)
        checks = aggregate.get("machine_checks", {})
        if (
            aggregate.get("status") != "pending_manual_review"
            or aggregate.get("machine_selection_decision") != "go"
            or set(checks) != MACHINE_CHECKS
            or not all(checks.values())
        ):
            self.halt("Pending final report is not a machine-validated GO")
        selection = aggregate.get("selection_report")
        manual = aggregate.get("manual_review", {})
        required_hash = manual.get("required_candidates_sha256")
        if not isinstance(required_hash, str) or len(required_hash) != 64:
            self.halt("Manual-review requirement digest is invalid")
        selection_path = Path(str(selection.get("path", ""))).resolve()
        if selection != attest(selection_path):
            self.halt("Selection report attestation changed before waiver")
        expected = {
            "schema_version": WAIVER_SCHEMA,
            "waive_manual_review": True,
            "production_plan": attest(self.plan),
            "selection_report": selection,
            "required_candidates_sha256": required_hash,
            "rationale": "The user explicitly authorized unattended completion and publication. Machine GO, every frozen machine check, exact provenance validation, cache audit, remote hash verification, and clean-room extraction remain mandatory; only manual listening is waived.",
            "authorization": self.config["authorization"],
        }
        if self.waiver.is_file():
            waiver = read_json(self.waiver)
            if {key: waiver.get(key) for key in expected} != expected:
                self.halt("Existing unattended waiver differs from the authorized scope")
            return selection_path
        immutable_write(self.waiver, json_bytes({**expected, "created_utc": utc_now()}))
        self.state("waiver_created", "explicit machine-GO-only manual-listening waiver", waiver=attest(self.waiver))
        return selection_path

    def run(self) -> None:
        self.state("queued", "waiting for guarded postprocess completion")
        self.wait_for_machine_result()
        selection = self.create_waiver()
        self.state("finalizing_go", "rerunning finalizer with explicit waiver")
        self.run_command(
            "finalize_go",
            [
                str(BASE_PYTHON),
                "training-data/qa_vivos_qwen_production_v6.py",
                "finalize",
                str(self.plan),
                "--selection-report",
                str(selection),
                "--qa-root",
                str(self.qa_root),
                "--out-dir",
                str(self.final_root),
                "--manual-waiver",
                str(self.waiver),
            ],
        )
        aggregate = read_json(self.final_root / "aggregate_report.json")
        if aggregate.get("status") != "go" or aggregate.get("manual_review", {}).get(
            "waiver"
        ) != attest(self.waiver):
            self.halt("Final report did not reach GO with the exact waiver")
        final_args = [
            str(self.plan),
            "--accepted",
            str(self.final_root / "accepted.jsonl"),
            "--selection",
            str(self.final_root / "selection.jsonl"),
            "--qa-report",
            str(self.final_root / "aggregate_report.json"),
        ]
        self.state("cache_preflight", "validating finalized provenance before Mimi")
        self.run_command(
            "cache_preflight",
            [str(BASE_PYTHON), "finetune/cache_vivos_full.py", "preflight", *final_args],
        )
        self.state("building_cache", "encoding accepted source and target audio with PyTorch Mimi")
        self.run_command(
            "cache_build",
            [
                str(BASE_PYTHON),
                "finetune/cache_vivos_full.py",
                "build",
                *final_args,
                "--dataset-root",
                str(self.dataset_root),
                "--gender-files",
                *(str(path) for path in self.gender_files),
                "--out-root",
                str(self.cache_root),
                "--device",
                "mps",
            ],
        )
        self.state("release_preflight", "independently auditing finalized Mimi cache")
        self.run_command(
            "release_preflight",
            [
                str(BASE_PYTHON),
                "finetune/release_vivos_cache.py",
                "preflight",
                *final_args,
                "--cache-root",
                str(self.cache_root),
            ],
        )
        self.state("preparing_release", "building immutable archives and metadata release")
        self.run_command(
            "release_prepare",
            [
                str(BASE_PYTHON),
                "finetune/release_vivos_cache.py",
                "prepare",
                *final_args,
                "--cache-root",
                str(self.cache_root),
            ],
        )
        self.state("publishing", "uploading fixed Hub prefix and running clean-room verification")
        self.run_command(
            "release_publish",
            [
                str(BASE_PYTHON),
                "finetune/release_vivos_cache.py",
                "publish",
                "--env-file",
                str(self.env_file),
            ],
        )
        report_path = self.release_dir / "release_report.json"
        report = read_json(report_path)
        if (
            report.get("repository") != "huybik/hibiki-zero-vi-full-sft"
            or report.get("remote_prefix") != "v2/vivos_qwen3_tts_mlx_retry_v6_full"
            or report.get("clean_room", {}).get("cache_audit_complete") is not True
        ):
            self.halt("Publication report does not prove the expected clean-room release")
        self.state(
            "published",
            "dataset release uploaded and clean-room verified",
            release_report=attest(report_path),
            commit_oid=report["commit_oid"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_plan", type=Path)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--supervisor-work", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gender-files", type=Path, nargs="+", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--poll-seconds", type=int, default=30, choices=range(1, 61))
    return parser.parse_args()


def main() -> None:
    completion: Completion | None = None
    try:
        completion = Completion(parse_args())
        completion.run()
    except Exception as error:
        if completion is not None:
            current = (
                read_json(completion.state_path).get("state")
                if completion.state_path.is_file()
                else None
            )
            if current != "halted":
                completion.state("halted", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

"""Guard and resume the frozen Qwen MLX retry-v6 postprocess state machine."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_retry_v6 import production_attestation_path
from synthesize_vivos import canonical_json, sha256_bytes, sha256_file


BASE_PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/bin/python")
MLX_PYTHON = Path("/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python")
QA_PYTHON = Path("/Volumes/data/envs/hibiki-vivos-qa/bin/python")
EXPECTED_ROWS = 10_950
EXPECTED_GROUPS = 1_391
ATTEMPTS = ("attempt0_t08", "retry1_t07", "retry2_t08")
SELECTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_selection_v6"
QA_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_qa_v6"
FINAL_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_final_v6"


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


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, (canonical_json(row) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = Path(__file__).resolve().parent.parent
        self.plan_path = args.production_plan.expanduser().resolve()
        self.production_root = self.plan_path.parent
        self.qa_root = args.qa_root.expanduser().resolve()
        self.work = args.work_dir.expanduser().resolve()
        self.final_root = self.qa_root / "final"
        self.attempt0_pid = args.attempt0_pid
        self.attempt0_session = args.attempt0_session
        self.supervisor_session = args.supervisor_session
        self.poll_seconds = args.poll_seconds
        self.exclude_speakers = sorted(args.exclude_speaker)
        self.events_path = self.work / "events.jsonl"
        self.history_path = self.work / "command_history.jsonl"
        self.state_path = self.work / "state.json"
        self.config_path = self.work / "config.json"
        self.work.mkdir(parents=True, exist_ok=True)
        self.lock_stream = (self.work / "supervisor.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another v6 postprocess supervisor holds the lock") from error
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> dict[str, Any]:
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        if plan.get("rows") != EXPECTED_ROWS or len(plan.get("groups", [])) != EXPECTED_GROUPS:
            raise RuntimeError("Supervisor requires the exact 10,950-row / 1,391-group plan")
        policy = Path(plan["policy"]["path"]).resolve()
        source = Path(plan["source_plan"]["path"]).resolve()
        production_attestation = production_attestation_path(self.plan_path)
        scripts = [
            Path(__file__).resolve(),
            self.repo / "training-data/benchmark_vivos_qwen_mlx_retry_v6.py",
            self.repo / "training-data/qwen_mlx_compaction.py",
            self.repo / "training-data/qwen_mlx_recurrent.py",
            self.repo / "training-data/validate_vivos_qwen_production_v6.py",
            self.repo / "training-data/qa_vivos_qwen_production_v6.py",
            self.repo / "training-data/run_vivos_qwen_production_v6.py",
            self.repo / "training-data/run_vivos_qwen_postprocess_v6.py",
        ]
        bindings = {
            "production_plan": attest(self.plan_path),
            "policy": attest(policy),
            "source_plan": attest(source),
            "production_attestation": attest(production_attestation),
            "scripts": [attest(path) for path in scripts],
        }
        stable = {
            "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_supervisor_config_v6",
            "production_root": str(self.production_root),
            "qa_root": str(self.qa_root),
            "final_root": str(self.final_root),
            "work_dir": str(self.work),
            "attempt0": {"pid": self.attempt0_pid, "session": self.attempt0_session},
            "supervisor_session": self.supervisor_session,
            "poll_seconds": self.poll_seconds,
            "speaker_exclusions": self.exclude_speakers,
            "expected_scope": {"rows": EXPECTED_ROWS, "groups": EXPECTED_GROUPS},
            "python": {
                "validator_selector_finalizer": str(BASE_PYTHON),
                "generation": str(MLX_PYTHON),
                "qa": str(QA_PYTHON),
            },
            "bindings": bindings,
        }
        if self.config_path.is_file():
            existing = json.loads(self.config_path.read_text(encoding="utf-8"))
            if {key: existing.get(key) for key in stable} != stable:
                raise RuntimeError("Existing supervisor config differs from current launch contract")
            return existing
        config = {**stable, "created_utc": utc_now()}
        immutable_write(self.config_path, json_bytes(config))
        return config

    def launch_record(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
        path = self.work / f"launch_{timestamp}.json"
        record = {
            "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_supervisor_launch_v6",
            "launched_utc": utc_now(),
            "pid": os.getpid(),
            "attempt0_alive": pid_alive(self.attempt0_pid),
            "attempt0_sentinel_present": self.attempt_manifest(0).is_file(),
            "config": attest(self.config_path),
            "config_bindings": self.config["bindings"],
        }
        immutable_write(path, json_bytes(record))
        return path

    def state(self, name: str, reason: str, **details: Any) -> None:
        previous = None
        if self.state_path.is_file():
            previous = json.loads(self.state_path.read_text(encoding="utf-8")).get("state")
        row = {
            "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_supervisor_event_v6",
            "utc": utc_now(),
            "previous_state": previous,
            "state": name,
            "reason": reason,
            "details": details,
        }
        append(self.events_path, row)
        atomic_write(
            self.state_path,
            json_bytes(
                {
                    "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_supervisor_state_v6",
                    "updated_utc": row["utc"],
                    "state": name,
                    "reason": reason,
                    "details": details,
                    "config": attest(self.config_path),
                    "events": str(self.events_path),
                }
            ),
        )

    def halt(self, reason: str, **details: Any) -> None:
        self.state("halted", reason, **details)
        raise RuntimeError(reason)

    def attempt_manifest(self, attempt: int) -> Path:
        return self.production_root / f"generation_attempt{attempt}_manifest.json"

    def validation_path(self, attempt: int) -> Path:
        return self.work / f"validation_attempt{attempt}.json"

    def retry_path(self, attempt: int) -> Path | None:
        return None if attempt == 0 else self.qa_root / f"retry_round{attempt}.jsonl"

    def qa_report_path(self, attempt: int) -> Path:
        return self.qa_root / ATTEMPTS[attempt] / "qa_report.json"

    def selection_path(self, attempt: int) -> Path:
        return self.qa_root / f"selection_round{attempt}.json"

    def _file_inputs(self, command: list[str]) -> list[dict[str, str]]:
        inputs = []
        for token in command[1:]:
            path = Path(token).expanduser()
            if not path.is_absolute():
                path = self.repo / path
            if path.is_file():
                record = attest(path)
                if record not in inputs:
                    inputs.append(record)
        return inputs

    def _completed_command(self, command_id: str) -> dict[str, Any] | None:
        if not self.history_path.is_file():
            return None
        completed = None
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("command_id") == command_id and row.get("event") == "completed":
                completed = row
        return completed

    def _unfinished_command(self, command_id: str) -> dict[str, Any] | None:
        if not self.history_path.is_file():
            return None
        pending = None
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("command_id") != command_id:
                continue
            if row.get("event") in {"started", "spawned"}:
                pending = row
            elif row.get("event") == "completed":
                pending = None
        return pending

    def run_logged(self, label: str, command: list[str]) -> tuple[int, Path]:
        inputs = self._file_inputs(command)
        command_id = sha256_bytes(
            canonical_json(
                {"label": label, "command": command, "cwd": str(self.repo), "file_inputs": inputs}
            ).encode()
        )
        completed = self._completed_command(command_id)
        if completed is not None and completed["returncode"] == 0:
            log = Path(completed["log"]["path"])
            if attest(log) != completed["log"]:
                self.halt("completed command log changed", label=label)
            return 0, log
        unfinished = self._unfinished_command(command_id)
        if unfinished is not None:
            child_pid = unfinished.get("child_pid")
            if child_pid and pid_alive(int(child_pid)):
                self.halt(
                    "an interrupted supervisor left this stage process alive",
                    label=label,
                    child_pid=child_pid,
                )
            if label.startswith("validate_attempt"):
                self.halt(
                    "validator execution was interrupted with unknown outcome; refusing to repeat it",
                    label=label,
                )
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
        log = self.work / "logs" / f"{timestamp}_{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = {
            "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_command_v6",
            "event": "started",
            "command_id": command_id,
            "label": label,
            "command": command,
            "cwd": str(self.repo),
            "file_inputs": inputs,
            "started_utc": utc_now(),
            "log_path": str(log),
        }
        append(self.history_path, started)
        started_monotonic = time.monotonic()
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            process = subprocess.Popen(
                command, cwd=self.repo, stdout=stream, stderr=subprocess.STDOUT
            )
            append(
                self.history_path,
                {**started, "event": "spawned", "spawned_utc": utc_now(), "child_pid": process.pid},
            )
            returncode = process.wait()
        record = {
            **started,
            "event": "completed",
            "completed_utc": utc_now(),
            "wall_seconds": time.monotonic() - started_monotonic,
            "returncode": returncode,
            "terminal_success": returncode == 0,
            "log": attest(log),
        }
        append(self.history_path, record)
        return returncode, log

    def _parse_json_log(self, log: Path) -> dict[str, Any]:
        try:
            value = json.loads(log.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.halt("command did not emit one machine-readable JSON object", log=str(log))
        if not isinstance(value, dict):
            self.halt("command JSON output was not an object", log=str(log))
        return value

    def wait_for_attempt0(self) -> None:
        if self.state_path.is_file():
            current = json.loads(self.state_path.read_text(encoding="utf-8"))
            if current.get("state") == "halted":
                raise RuntimeError("Supervisor is halted; refusing automatic restart")
        if self.validation_path(0).is_file():
            return
        self.state(
            "waiting_attempt0",
            "polling only completion sentinel and attested generator PID",
            pid=self.attempt0_pid,
            sentinel=str(self.attempt_manifest(0)),
        )
        waiting_for_exit = False
        while pid_alive(self.attempt0_pid):
            if self.attempt_manifest(0).is_file():
                if not waiting_for_exit:
                    self.state(
                        "waiting_attempt0_exit",
                        "completion sentinel exists; waiting for MLX generator PID to exit before QA",
                        pid=self.attempt0_pid,
                    )
                    waiting_for_exit = True
            time.sleep(self.poll_seconds)

    def validate_generation(self, attempt: int) -> dict[str, Any]:
        output = self.validation_path(attempt)
        if output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
        else:
            command = [
                str(BASE_PYTHON),
                "training-data/validate_vivos_qwen_production_v6.py",
                "production",
                str(self.plan_path),
                "--attempt",
                str(attempt),
            ]
            retry = self.retry_path(attempt)
            if retry is not None:
                command.extend(["--retry-manifest", str(retry)])
            self.state(f"validating_attempt{attempt}", "running exact CPU completion validator once")
            returncode, log = self.run_logged(f"validate_attempt{attempt}", command)
            result = self._parse_json_log(log)
            immutable_write(output, json_bytes(result))
            if returncode:
                self.halt(
                    "generation validator rejected attempt",
                    attempt=attempt,
                    returncode=returncode,
                    validation=attest(output),
                )
        expected_rows = EXPECTED_ROWS if attempt == 0 else len(
            self.retry_path(attempt).read_text(encoding="utf-8").splitlines()
        )
        if (
            result.get("state") != "complete"
            or result.get("expected_rows") != expected_rows
            or result.get("completed_rows") != expected_rows
            or (attempt == 0 and result.get("expected_groups") != EXPECTED_GROUPS)
            or result.get("expected_groups") != result.get("completed_groups")
            or result.get("media_error_rows", 0) != 0
        ):
            self.halt("generation did not validate as exact and complete", attempt=attempt, result=result)
        return result

    def qa_complete(self, attempt: int, expected_rows: int) -> bool:
        report_path = self.qa_report_path(attempt)
        metrics = report_path.parent / "metrics.jsonl"
        if not report_path.is_file() or not metrics.is_file():
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != QA_SCHEMA
            or report.get("status") != "complete"
            or report.get("attempt") != attempt
            or report.get("scope_rows") != expected_rows
            or report.get("production_plan") != attest(self.plan_path)
            or report.get("generation_manifest") != attest(self.attempt_manifest(attempt))
            or report.get("row_metrics") != attest(metrics)
        ):
            self.halt("existing QA artifact has an invalid scope binding", attempt=attempt)
        return True

    def score(self, attempt: int, expected_rows: int) -> None:
        if self.qa_complete(attempt, expected_rows):
            return
        if attempt == 0 and pid_alive(self.attempt0_pid):
            self.halt("refusing to overlap attempt-0 MLX generation and PyTorch MPS QA")
        command = [
            str(QA_PYTHON),
            "training-data/qa_vivos_qwen_production_v6.py",
            "score-attempt",
            str(self.plan_path),
            "--attempt",
            str(attempt),
            "--out-dir",
            str(self.qa_report_path(attempt).parent),
            "--device",
            "mps",
        ]
        retry = self.retry_path(attempt)
        if retry is not None:
            command.extend(["--retry-manifest", str(retry)])
        self.state(f"scoring_attempt{attempt}", "running pinned resumable MPS QA", rows=expected_rows)
        returncode, _ = self.run_logged(f"score_attempt{attempt}", command)
        if returncode or not self.qa_complete(attempt, expected_rows):
            self.halt("QA did not produce an exact complete report", attempt=attempt, returncode=returncode)

    def selection(self, through_round: int) -> dict[str, Any]:
        path = self.selection_path(through_round)
        if not path.is_file():
            command = [
                str(BASE_PYTHON),
                "training-data/qa_vivos_qwen_production_v6.py",
                "select",
                str(self.plan_path),
                "--through-round",
                str(through_round),
                "--qa-root",
                str(self.qa_root),
                "--out-dir",
                str(self.qa_root),
            ]
            for speaker_id in self.exclude_speakers:
                command.extend(["--exclude-speaker", speaker_id])
            self.state(f"selecting_round{through_round}", "running frozen v6 selector")
            returncode, _ = self.run_logged(f"select_round{through_round}", command)
            if returncode or not path.is_file():
                self.halt("selector did not produce its report", round=through_round)
        report = json.loads(path.read_text(encoding="utf-8"))
        selection_rows = Path(str(report.get("selection_rows", {}).get("path", "")))
        if (
            report.get("schema_version") != SELECTION_SCHEMA
            or report.get("production_plan") != attest(self.plan_path)
            or report.get("through_round") != through_round
            or report.get("speaker_exclusions", {}).get("speaker_ids")
            != self.exclude_speakers
            or report.get("decision") not in {"go", "continue", "no_go"}
            or not selection_rows.is_file()
            or report.get("selection_rows") != attest(selection_rows)
        ):
            self.halt("selection report has an invalid scope binding", round=through_round)
        if report["decision"] == "continue":
            retry = self.retry_path(through_round + 1)
            if retry is None or report.get("next_retry") != attest(retry):
                self.halt("continue decision lacks the exact immutable retry manifest")
        elif report.get("next_retry") is not None:
            self.halt("terminal selection unexpectedly names another retry")
        return report

    def generate_retry(self, attempt: int) -> dict[str, Any]:
        validation = self.validation_path(attempt)
        if validation.is_file():
            return self.validate_generation(attempt)
        retry = self.retry_path(attempt)
        assert retry is not None
        command = [
            str(MLX_PYTHON),
            "training-data/run_vivos_qwen_production_v6.py",
            "run",
            str(self.plan_path),
            "--round",
            str(attempt),
            "--retry-ids",
            str(retry),
        ]
        self.state(f"generating_attempt{attempt}", "running exact pinned MLX retry runner")
        returncode, _ = self.run_logged(f"generate_attempt{attempt}", command)
        result = self.validate_generation(attempt)
        if returncode:
            self.halt("MLX retry command failed", attempt=attempt, returncode=returncode)
        return result

    def finalize(self, report: dict[str, Any]) -> None:
        aggregate = self.final_root / "aggregate_report.json"
        if not aggregate.is_file():
            selection = self.selection_path(int(report["through_round"]))
            command = [
                str(BASE_PYTHON),
                "training-data/qa_vivos_qwen_production_v6.py",
                "finalize",
                str(self.plan_path),
                "--selection-report",
                str(selection),
                "--qa-root",
                str(self.qa_root),
                "--out-dir",
                str(self.final_root),
            ]
            self.state("finalizing", "materializing terminal selection without a review waiver")
            returncode, _ = self.run_logged("finalize", command)
            if returncode or not aggregate.is_file():
                self.halt("finalizer did not produce an aggregate report", returncode=returncode)
        final = json.loads(aggregate.read_text(encoding="utf-8"))
        if (
            final.get("schema_version") != FINAL_SCHEMA
            or final.get("production_plan") != attest(self.plan_path)
            or final.get("selection_report")
            != attest(self.selection_path(int(report["through_round"])))
            or final.get("status") not in {"pending_manual_review", "no_go"}
        ):
            self.halt("final aggregate report has an invalid terminal binding")
        self.state(
            final["status"],
            "postprocess stopped before cache/package/publication",
            aggregate_report=attest(aggregate),
        )

    def run(self) -> None:
        self.launch_record()
        self.wait_for_attempt0()
        validation = self.validate_generation(0)
        self.score(0, int(validation["expected_rows"]))
        report = self.selection(0)
        for attempt in (1, 2):
            if report["decision"] != "continue":
                break
            validation = self.generate_retry(attempt)
            self.score(attempt, int(validation["expected_rows"]))
            report = self.selection(attempt)
        if report["decision"] == "continue":
            self.halt("selector requested a retry beyond the frozen two-round maximum")
        self.finalize(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_plan", type=Path)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--attempt0-pid", type=int, required=True)
    parser.add_argument("--attempt0-session", required=True)
    parser.add_argument("--supervisor-session", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30, choices=range(1, 61))
    parser.add_argument("--exclude-speaker", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor(args)
        supervisor.run()
    except Exception as error:
        if supervisor is not None:
            current = (
                json.loads(supervisor.state_path.read_text(encoding="utf-8")).get("state")
                if supervisor.state_path.is_file()
                else None
            )
            if current != "halted":
                supervisor.state("halted", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

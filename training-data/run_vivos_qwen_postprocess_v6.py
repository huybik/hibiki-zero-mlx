"""Run one v6 postprocess command with append-only command and timing records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from synthesize_vivos import canonical_json, sha256_bytes, sha256_file


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, (canonical_json(row) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("one command is required after --")
    allowed_scripts = {
        "validate_vivos_qwen_production_v6.py",
        "qa_vivos_qwen_production_v6.py",
        "run_vivos_qwen_production_v6.py",
    }
    if (
        len(command) < 2
        or "python" not in Path(command[0]).name
        or Path(command[1]).name not in allowed_scripts
    ):
        parser.error(
            "command must be one direct Python invocation of an approved v6 pipeline script"
        )
    log_dir = args.log_dir.expanduser().resolve()
    history_path = log_dir / "command_history.jsonl"
    cwd = Path.cwd().resolve()
    file_inputs = []
    for token in command[1:]:
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if path.is_file():
            file_inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    command_id = sha256_bytes(
        canonical_json({"command": command, "cwd": str(cwd), "file_inputs": file_inputs}).encode()
    )
    if history_path.is_file() and not args.repeat:
        for row in (
            json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
        ):
            if row.get("command_id") == command_id and row.get("terminal_success") is True:
                print(f"Already completed successfully: {command_id}")
                return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in args.label
    )
    log_path = log_dir / f"{timestamp}_{safe_label}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    append(
        history_path,
        {
            "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_command_v6",
            "event": "started",
            "command_id": command_id,
            "label": args.label,
            "command": command,
            "cwd": str(cwd),
            "file_inputs": file_inputs,
            "started_utc": started_utc,
            "log_path": str(log_path),
        },
    )
    with os.fdopen(descriptor, "wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    tail = log_path.read_bytes()[-1_000_000:]
    nonterminal = any(
        marker in tail
        for marker in (
            b'"state": "incomplete"',
            b'"status": "pending_manual_review"',
        )
    )
    record = {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_postprocess_command_v6",
        "event": "completed",
        "command_id": command_id,
        "label": args.label,
        "command": command,
        "cwd": str(cwd),
        "file_inputs": file_inputs,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "returncode": result.returncode,
        "terminal_success": result.returncode == 0 and not nonterminal,
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
    }
    append(history_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

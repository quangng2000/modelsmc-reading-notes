"""Operational scheduler for the frozen 36-run developmental SMC matrix.

This file does not define experiment semantics.  Every child invocation passes
through ``run_developmental_smc_benchmark``, which revalidates the externally
frozen protocol, source tree, task bytes, and exact command.  The scheduler
records its own source hash and inter-task concurrency before launching a call.
Failed child runs are retained and are never retried or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.run_developmental_smc_benchmark import ARMS, TASK_IDS

FROZEN_PROTOCOL_SHA256 = (
    "0b7092349c5e240aac378bd04a97020ea15e170d56b8d573890d418217db689d"
)
EXECUTION_ORDER = ("evidence-only", "grammar-only", "llm-smc")
INTER_TASK_CONCURRENCY = {
    "evidence-only": 6,
    "grammar-only": 6,
    "llm-smc": 4,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")


def _run_one(
    *,
    repo_root: Path,
    protocol: Path,
    suite_dir: Path,
    runs_root: Path,
    logs_root: Path,
    task_id: str,
    arm: str,
    python_executable: str,
) -> dict[str, object]:
    command = [
        python_executable,
        "-m",
        "research.run_developmental_smc_benchmark",
        "--repo-root",
        str(repo_root),
        "--protocol",
        str(protocol),
        "--expected-protocol-sha256",
        FROZEN_PROTOCOL_SHA256,
        "--suite-dir",
        str(suite_dir),
        "--runs-root",
        str(runs_root),
        "--task-id",
        task_id,
        "--arm",
        arm,
        "--python-executable",
        python_executable,
    ]
    log_path = logs_root / arm / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "task_id": task_id,
        "arm": arm,
        "returncode": completed.returncode,
        "started_unix": started,
        "completed_unix": time.time(),
        "log_path": str(log_path),
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    protocol = args.protocol.resolve()
    suite_dir = args.suite_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matrix output: {output}")
    if _sha256_file(protocol) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("matrix protocol differs from the frozen external digest")
    if tuple(ARMS) != ("llm-smc", "evidence-only", "grammar-only"):
        raise ValueError("runner arm contract drifted")
    if len(TASK_IDS) != 12:
        raise ValueError("runner task contract drifted")

    runs_root = output / "runs"
    logs_root = output / "driver-logs"
    output.mkdir(parents=True)
    launch = {
        "schema": "developmental-smc-matrix-launch-v1",
        "classification": "developmental-nonblind",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "scheduler_sha256": _sha256_file(Path(__file__)),
        "execution_order": list(EXECUTION_ORDER),
        "task_order": list(TASK_IDS),
        "inter_task_concurrency": INTER_TASK_CONCURRENCY,
        "retry_policy": "none; every failed child is retained",
        "child_count": len(TASK_IDS) * len(EXECUTION_ORDER),
        "started_unix": time.time(),
    }
    _write_exclusive(output / "launch-seal.json", launch)

    records: list[dict[str, object]] = []
    for arm in EXECUTION_ORDER:
        with ThreadPoolExecutor(max_workers=INTER_TASK_CONCURRENCY[arm]) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    repo_root=repo_root,
                    protocol=protocol,
                    suite_dir=suite_dir,
                    runs_root=runs_root,
                    logs_root=logs_root,
                    task_id=task_id,
                    arm=arm,
                    python_executable=args.python_executable,
                ): task_id
                for task_id in TASK_IDS
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    ordered = sorted(
        records,
        key=lambda record: (
            EXECUTION_ORDER.index(str(record["arm"])),
            TASK_IDS.index(str(record["task_id"])),
        ),
    )
    summary = {
        "schema": "developmental-smc-matrix-driver-result-v1",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "completed_unix": time.time(),
        "successful_children": sum(record["returncode"] == 0 for record in ordered),
        "failed_children": sum(record["returncode"] != 0 for record in ordered),
        "children": ordered,
    }
    _write_exclusive(output / "driver-result.json", summary)
    return 0 if summary["failed_children"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

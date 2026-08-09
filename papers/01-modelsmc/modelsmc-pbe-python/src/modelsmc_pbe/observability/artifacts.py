"""Durable file layout for one experiment run."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from modelsmc_pbe.observability.runtime_metadata import git_state, runtime_snapshot, slug
from modelsmc_pbe.observability.serialization import jsonable, write_json_atomic
from modelsmc_pbe.runtime.device import DeviceInfo

SCHEMA_VERSION = 1


@dataclass(slots=True)
class RunArtifacts:
    """Own the stable on-disk schema for one run."""

    run_dir: Path
    run_id: str
    manifest: dict[str, Any]
    include_raw_payloads: bool

    @classmethod
    def create(
        cls,
        *,
        base_dir: str | Path,
        run_name: str,
        config: Any,
        device: DeviceInfo,
        seed: int,
        probabilistic_claim: str,
        include_raw_payloads: bool,
        run_id: str | None,
        cwd: str | Path | None,
    ) -> RunArtifacts:
        """Create a unique directory and its initial reproducibility manifest."""

        identifier = run_id or uuid.uuid4().hex
        working_directory = Path(cwd or Path.cwd()).resolve()
        timestamp = datetime.now(UTC)
        directory_name = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{slug(run_name)}-{identifier[:8]}"
        run_dir = Path(base_dir).expanduser().resolve() / directory_name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(f"run artifact directory already exists: {run_dir}") from error

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": identifier,
            "name": run_name,
            "status": "running",
            "started_at": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "completed_at": None,
            "command": list(sys.argv),
            "working_directory": str(working_directory),
            "configuration": jsonable(config, include_raw_payloads=include_raw_payloads),
            "device": device.as_dict(),
            "seed": seed,
            "probabilistic_claim": probabilistic_claim,
            "git": git_state(working_directory),
            "runtime": runtime_snapshot(),
            "artifacts": {
                "events": "events.jsonl",
                "result": "result.json",
                "final_particles": "final_particles.jsonl",
            },
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
        return cls(
            run_dir=run_dir,
            run_id=identifier,
            manifest=manifest,
            include_raw_payloads=include_raw_payloads,
        )

    def open_event_stream(self) -> TextIO:
        """Open the append-only event stream with line buffering."""

        return (self.run_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def write_final_particles(self, particles: Iterable[Any]) -> int:
        """Atomically write final particle records and return their count."""

        destination = self.run_dir / "final_particles.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        count = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for rank, particle in enumerate(particles):
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "rank": rank,
                    "particle": jsonable(particle, include_raw_payloads=self.include_raw_payloads),
                }
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                stream.write("\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return count

    def write_completed_result(
        self, *, result: Any, particle_count: int, completed_at: str
    ) -> None:
        """Persist the stable success-result envelope."""

        write_json_atomic(
            self.run_dir / "result.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": "completed",
                "completed_at": completed_at,
                "particle_count": particle_count,
                "result": jsonable(result, include_raw_payloads=self.include_raw_payloads),
            },
        )

    def write_failed_result(self, *, failure: dict[str, Any], completed_at: str) -> None:
        """Persist the stable failure-result envelope."""

        write_json_atomic(
            self.run_dir / "result.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": "failed",
                "completed_at": completed_at,
                "error": failure,
            },
        )

    def seal_manifest(self, *, status: str, completed_at: str) -> None:
        """Mark the manifest terminal only after all other artifacts exist."""

        self.manifest["status"] = status
        self.manifest["completed_at"] = completed_at
        write_json_atomic(self.run_dir / "manifest.json", self.manifest)

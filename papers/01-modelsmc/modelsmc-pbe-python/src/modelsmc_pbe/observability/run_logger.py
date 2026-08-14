"""Console/event lifecycle for one observable experiment run."""

from __future__ import annotations

import json
import threading
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Any, Literal, TextIO

from rich.console import Console

from modelsmc_pbe.observability.artifacts import SCHEMA_VERSION, RunArtifacts
from modelsmc_pbe.observability.console_renderer import EVENT_LEVELS, ConsoleEventRenderer
from modelsmc_pbe.observability.serialization import iso_now, jsonable
from modelsmc_pbe.runtime.device import DeviceInfo


class RunLogger:
    """Append-only event logger and owner of one research run directory."""

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        manifest: dict[str, Any],
        console_level: str,
        include_raw_payloads: bool,
        console: Console | None,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self._manifest = manifest
        self._console_level = console_level
        self._include_raw_payloads = include_raw_payloads
        self._console = console or Console(stderr=True)
        self._renderer = ConsoleEventRenderer(console_level, self._console)
        self._artifacts = RunArtifacts(
            run_dir=run_dir,
            run_id=run_id,
            manifest=manifest,
            include_raw_payloads=include_raw_payloads,
        )
        self._events: TextIO = self._artifacts.open_event_stream()
        self._lock = threading.RLock()
        self._sequence = 0
        self._started_monotonic = monotonic()
        self._finished = False
        self.event("run.started", message="run started", level="info")

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
        console_level: str = "info",
        include_raw_payloads: bool = False,
        run_id: str | None = None,
        cwd: str | Path | None = None,
        console: Console | None = None,
    ) -> RunLogger:
        """Create a unique run directory and write its initial manifest."""

        artifacts = RunArtifacts.create(
            base_dir=base_dir,
            run_name=run_name,
            config=config,
            device=device,
            seed=seed,
            probabilistic_claim=probabilistic_claim,
            include_raw_payloads=include_raw_payloads,
            run_id=run_id,
            cwd=cwd,
        )
        return cls(
            run_dir=artifacts.run_dir,
            run_id=artifacts.run_id,
            manifest=artifacts.manifest,
            console_level=console_level,
            include_raw_payloads=include_raw_payloads,
            console=console,
        )

    def _show_console(
        self,
        level: str,
        event: str,
        message: str | None,
        data: Mapping[str, Any],
    ) -> None:
        self._renderer.render(level=level, event=event, message=message, data=data)

    def event(
        self,
        event: str,
        *,
        message: str | None = None,
        level: str = "info",
        **data: Any,
    ) -> dict[str, Any]:
        """Append and flush one machine-readable event, then render it for humans."""

        if level not in EVENT_LEVELS or level == "quiet":
            raise ValueError("event level must be trace, debug, info, warning, or error")
        with self._lock:
            if self._finished:
                raise RuntimeError("cannot emit events after the run is finished")
            sanitized = jsonable(data, include_raw_payloads=self._include_raw_payloads)
            record = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp": iso_now(),
                "elapsed_seconds": round(monotonic() - self._started_monotonic, 6),
                "level": level,
                "event": event,
                "message": message,
                "data": sanitized,
            }
            self._sequence += 1
            self._events.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            self._events.write("\n")
            self._events.flush()
            self._show_console(level, event, message, sanitized)
            return record

    def write_final_particles(self, particles: Iterable[Any]) -> int:
        """Atomically write final particle records and return their count."""

        return self._artifacts.write_final_particles(particles)

    def record_metrics(self, name: str, metrics: Any) -> None:
        """Persist one named run-level metric summary in the sealed manifest."""

        if not name.strip():
            raise ValueError("metric summary name must not be empty")
        with self._lock:
            if self._finished:
                raise RuntimeError("cannot record metrics after the run is finished")
            summaries = self._manifest.setdefault("metrics", {})
            if not isinstance(summaries, dict):  # pragma: no cover - manifest invariant
                raise RuntimeError("manifest metrics field is not an object")
            if name in summaries:
                raise ValueError(f"metric summary {name!r} was already recorded")
            summaries[name] = jsonable(
                metrics, include_raw_payloads=self._include_raw_payloads
            )

    def finish(self, *, result: Any, final_particles: Iterable[Any]) -> None:
        """Write final artifacts and seal a successful run."""

        with self._lock:
            if self._finished:
                raise RuntimeError("run is already finished")
            particle_count = self.write_final_particles(final_particles)
            completed_at = iso_now()
            self._artifacts.write_completed_result(
                result=result,
                particle_count=particle_count,
                completed_at=completed_at,
            )
            self.event(
                "run.completed",
                message="run completed",
                level="info",
                particle_count=particle_count,
            )
            self._seal(status="completed", completed_at=completed_at)

    def fail(self, error: BaseException) -> None:
        """Persist a failed result and seal the run before re-raising upstream."""

        with self._lock:
            if self._finished:
                return
            completed_at = iso_now()
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
            self._artifacts.write_failed_result(failure=failure, completed_at=completed_at)
            # Keep the artifact set predictable even when no final population exists.
            self.write_final_particles(())
            self.event("run.failed", message=str(error), level="error", error=failure)
            self._seal(status="failed", completed_at=completed_at)

    def _seal(self, *, status: str, completed_at: str) -> None:
        self._artifacts.seal_manifest(status=status, completed_at=completed_at)
        self._events.close()
        self._finished = True

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> Literal[False]:
        del error_type, traceback_value
        if error is not None:
            self.fail(error)
        elif not self._finished:
            self.finish(result={}, final_particles=())
        return False

"""Strict loader for the frozen benchmark protocol.

The research harness deliberately uses the standard library.  A protocol is
data, not executable configuration: every command-line flag is assembled from
an allowlist below, and provider credentials are referenced only by variable
name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

AnalysisLabel = Literal["exploratory", "confirmatory"]
ArmName = Literal["U", "D", "Q", "QD"]


@dataclass(frozen=True, slots=True)
class HeldoutSpec:
    oracle: str
    seed: int
    count: int
    minimum: int
    maximum: int
    max_length: int | None
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    spec_path: Path
    label: AnalysisLabel
    seen_during_development: bool
    heldout: HeldoutSpec


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: ArmName
    proposal: Literal["catalog", "vllm"]
    deduction_mix: float
    description: str


@dataclass(frozen=True, slots=True)
class Caps:
    particles: int
    iterations: int
    alpha: float
    ess_threshold: float
    max_scored_candidates: int
    support_limit: int
    hole_state_limit: int
    hole_max_cost: int
    wall_time_seconds: float
    provider_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    model: str
    base_url_env: str
    api_key_env: str | None


@dataclass(frozen=True, slots=True)
class Protocol:
    path: Path
    protocol_id: str
    schema_version: int
    status: str
    protocol_sha256: str
    seeds: tuple[int, ...]
    tasks: tuple[TaskSpec, ...]
    arms: tuple[ArmSpec, ...]
    caps: Caps
    provider: ProviderSpec
    shared_arguments: dict[str, str | int | float]

    @property
    def project_root(self) -> Path:
        return self.path.parent.parent


def _record(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _load_task(value: object, *, protocol_dir: Path) -> TaskSpec:
    record = _record(value, "task")
    task_id = _text(record.get("id"), "task.id")
    label = _text(record.get("analysis_label"), f"task {task_id} analysis_label")
    if label not in {"exploratory", "confirmatory"}:
        raise ValueError(f"task {task_id} has invalid analysis_label {label!r}")
    seen = record.get("seen_during_development")
    if not isinstance(seen, bool):
        raise ValueError(f"task {task_id} seen_during_development must be boolean")
    spec_path = (protocol_dir / _text(record.get("spec"), f"task {task_id} spec")).resolve()
    if not spec_path.is_file():
        raise ValueError(f"task {task_id} specification does not exist: {spec_path}")
    heldout = _record(record.get("heldout"), f"task {task_id} heldout")
    minimum = _integer(heldout.get("minimum"), f"task {task_id} heldout.minimum")
    maximum = _integer(heldout.get("maximum"), f"task {task_id} heldout.maximum")
    if maximum < minimum:
        raise ValueError(f"task {task_id} heldout maximum precedes minimum")
    raw_max_length = heldout.get("max_length")
    max_length = (
        None
        if raw_max_length is None
        else _integer(raw_max_length, f"task {task_id} heldout.max_length", minimum=0)
    )
    return TaskSpec(
        task_id=task_id,
        spec_path=spec_path,
        label=cast(AnalysisLabel, label),
        seen_during_development=seen,
        heldout=HeldoutSpec(
            oracle=_text(heldout.get("oracle"), f"task {task_id} heldout.oracle"),
            seed=_integer(heldout.get("seed"), f"task {task_id} heldout.seed", minimum=0),
            count=_integer(heldout.get("count"), f"task {task_id} heldout.count", minimum=1),
            minimum=minimum,
            maximum=maximum,
            max_length=max_length,
            corpus_sha256=_text(
                heldout.get("corpus_sha256"), f"task {task_id} heldout.corpus_sha256"
            ),
        ),
    )


def _load_arm(value: object) -> ArmSpec:
    record = _record(value, "arm")
    name = _text(record.get("id"), "arm.id")
    if name not in {"U", "D", "Q", "QD"}:
        raise ValueError(f"unknown arm {name!r}")
    proposal = _text(record.get("proposal"), f"arm {name} proposal")
    if proposal not in {"catalog", "vllm"}:
        raise ValueError(f"arm {name} has invalid proposal {proposal!r}")
    deduction_mix = _number(record.get("deduction_mix"), f"arm {name} deduction_mix")
    if deduction_mix > 1.0:
        raise ValueError(f"arm {name} deduction_mix must be at most 1")
    return ArmSpec(
        name=cast(ArmName, name),
        proposal=cast(Literal["catalog", "vllm"], proposal),
        deduction_mix=deduction_mix,
        description=_text(record.get("description"), f"arm {name} description"),
    )


def load_protocol(path: str | Path) -> Protocol:
    """Load and validate one protocol, preserving a hash of its exact bytes."""

    resolved = Path(path).expanduser().resolve()
    content = resolved.read_bytes()
    try:
        document = _record(json.loads(content), "protocol")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid protocol JSON: {error}") from error
    protocol_id = _text(document.get("protocol_id"), "protocol_id")
    seeds = tuple(
        _integer(seed, "seed", minimum=0) for seed in _sequence(document.get("seeds"), "seeds")
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique array")
    tasks = tuple(
        _load_task(task, protocol_dir=resolved.parent)
        for task in _sequence(document.get("tasks"), "tasks")
    )
    arms = tuple(_load_arm(arm) for arm in _sequence(document.get("arms"), "arms"))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique")
    if {arm.name for arm in arms} != {"U", "D", "Q", "QD"}:
        raise ValueError("protocol must declare exactly the U, D, Q, and QD arms")

    caps_record = _record(document.get("caps"), "caps")
    caps = Caps(
        particles=_integer(caps_record.get("particles"), "caps.particles", minimum=1),
        iterations=_integer(caps_record.get("iterations"), "caps.iterations", minimum=1),
        alpha=_number(caps_record.get("alpha"), "caps.alpha"),
        ess_threshold=_number(caps_record.get("ess_threshold"), "caps.ess_threshold"),
        max_scored_candidates=_integer(
            caps_record.get("max_scored_candidates"), "caps.max_scored_candidates", minimum=1
        ),
        support_limit=_integer(
            caps_record.get("support_limit"), "caps.support_limit", minimum=1
        ),
        hole_state_limit=_integer(
            caps_record.get("hole_state_limit"), "caps.hole_state_limit", minimum=1
        ),
        hole_max_cost=_integer(
            caps_record.get("hole_max_cost"), "caps.hole_max_cost", minimum=1
        ),
        wall_time_seconds=_number(
            caps_record.get("wall_time_seconds"), "caps.wall_time_seconds", minimum=0.1
        ),
        provider_timeout_seconds=_number(
            caps_record.get("provider_timeout_seconds"),
            "caps.provider_timeout_seconds",
            minimum=0.1,
        ),
    )
    if caps.alpha > 1 or caps.ess_threshold > 1:
        raise ValueError("caps.alpha and caps.ess_threshold must be at most 1")

    provider_record = _record(document.get("provider"), "provider")
    raw_api_env = provider_record.get("api_key_env")
    if raw_api_env is not None and not isinstance(raw_api_env, str):
        raise ValueError("provider.api_key_env must be null or a string")
    provider = ProviderSpec(
        model=_text(provider_record.get("model"), "provider.model"),
        base_url_env=_text(provider_record.get("base_url_env"), "provider.base_url_env"),
        api_key_env=raw_api_env,
    )
    shared = _record(document.get("shared_arguments"), "shared_arguments")
    allowed_shared = {
        "beta_max",
        "candidate_batch_size",
        "deduction_strength",
        "proposal_epsilon",
        "temperature",
    }
    unexpected = set(shared) - allowed_shared
    if unexpected:
        raise ValueError(f"unsupported shared_arguments: {sorted(unexpected)}")
    if any(not isinstance(value, (str, int, float)) for value in shared.values()):
        raise ValueError("shared argument values must be scalar")

    return Protocol(
        path=resolved,
        protocol_id=protocol_id,
        schema_version=_integer(document.get("schema_version"), "schema_version", minimum=1),
        status=_text(document.get("status"), "status"),
        protocol_sha256=hashlib.sha256(content).hexdigest(),
        seeds=seeds,
        tasks=tasks,
        arms=arms,
        caps=caps,
        provider=provider,
        shared_arguments=cast(dict[str, str | int | float], shared),
    )

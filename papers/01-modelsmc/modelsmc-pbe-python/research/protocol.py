"""Strict loader for the frozen benchmark protocol.

The research harness deliberately uses the standard library.  A protocol is
data, not executable configuration: every command-line flag is assembled from
an allowlist below, and provider credentials are referenced only by variable
name.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    family_deduction_mix: float | None
    hole_deduction_mix: float | None
    description: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One immutable checkpoint in an ordered within-family size comparison."""

    model_id: str
    alias: str
    hf_repository: str
    architecture: str
    parameterization: Literal["dense", "mixture-of-experts"]
    total_parameters_billion: float
    active_parameters_billion: float
    dtype: str
    quantization: str
    model_revision: str
    tokenizer_revision: str
    base_url_env: str | None


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
    # ``model`` is retained only for schema-v1 compatibility.  Schema-v2
    # protocols declare immutable checkpoints in ``models`` instead.
    model: str | None
    base_url_env: str
    api_key_env: str | None
    vllm_server_config: str | None
    score_cache_dir: Path | None
    score_cache_mode: Literal["off", "read-write", "replay-only"]


@dataclass(frozen=True, slots=True)
class StageCaps:
    particles: int | None = None
    iterations: int | None = None
    max_scored_candidates: int | None = None
    wall_time_seconds: float | None = None
    provider_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    description: str
    task_ids: tuple[str, ...]
    arms: tuple[ArmName, ...]
    seeds: tuple[int, ...]
    model_ids: tuple[str, ...]
    caps: StageCaps
    max_provider_scored_candidates: int
    requires_audit_stage: str | None


@dataclass(frozen=True, slots=True)
class TargetContract:
    base_measure: Literal["equal-family-within-family-occam"]
    likelihood: Literal["soft-loss-gibbs"]
    loss_scale: float
    beta_max: float
    terminal_dominance_requirement: Literal[
        "min-exact-log-target-strictly-greater-than-max-inexact-log-target"
    ]
    reference_audit_required: bool
    reference_audit_stage: str


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
    models: tuple[ModelSpec, ...]
    stages: tuple[StageSpec, ...]
    caps: Caps
    provider: ProviderSpec
    materialize_reference: bool
    shared_arguments: dict[str, str | int | float]
    target_contract: TargetContract | None

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


_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _stable_id(value: object, name: str) -> str:
    result = _text(value, name)
    if _STABLE_ID.fullmatch(result) is None:
        raise ValueError(f"{name} must match {_STABLE_ID.pattern}")
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
    optional_mixes: dict[str, float | None] = {}
    for field_name in ("family_deduction_mix", "hole_deduction_mix"):
        raw_value = record.get(field_name)
        if raw_value is None:
            optional_mixes[field_name] = None
            continue
        parsed = _number(raw_value, f"arm {name} {field_name}")
        if parsed > 1.0:
            raise ValueError(f"arm {name} {field_name} must be at most 1")
        optional_mixes[field_name] = parsed
    return ArmSpec(
        name=cast(ArmName, name),
        proposal=cast(Literal["catalog", "vllm"], proposal),
        deduction_mix=deduction_mix,
        family_deduction_mix=optional_mixes["family_deduction_mix"],
        hole_deduction_mix=optional_mixes["hole_deduction_mix"],
        description=_text(record.get("description"), f"arm {name} description"),
    )


def _load_model(value: object) -> ModelSpec:
    record = _record(value, "model")
    model_id = _stable_id(record.get("id"), "model.id")
    parameterization = _text(
        record.get("parameterization"), f"model {model_id} parameterization"
    )
    if parameterization not in {"dense", "mixture-of-experts"}:
        raise ValueError(
            f"model {model_id} parameterization must be dense or mixture-of-experts"
        )
    total = _number(
        record.get("total_parameters_billion"),
        f"model {model_id} total_parameters_billion",
        minimum=0.000001,
    )
    active = _number(
        record.get("active_parameters_billion"),
        f"model {model_id} active_parameters_billion",
        minimum=0.000001,
    )
    if active > total:
        raise ValueError(f"model {model_id} active parameter count exceeds total")
    if parameterization == "dense" and active != total:
        raise ValueError(f"dense model {model_id} must have equal total and active size")
    raw_endpoint_env = record.get("base_url_env")
    if raw_endpoint_env is not None and not isinstance(raw_endpoint_env, str):
        raise ValueError(f"model {model_id} base_url_env must be null or a string")
    model_revision = _text(
        record.get("model_revision"), f"model {model_id} model_revision"
    )
    tokenizer_revision = _text(
        record.get("tokenizer_revision"), f"model {model_id} tokenizer_revision"
    )
    immutable_revision = re.compile(r"^[0-9a-f]{40}$")
    if immutable_revision.fullmatch(model_revision) is None:
        raise ValueError(f"model {model_id} model_revision must be a 40-hex commit")
    if immutable_revision.fullmatch(tokenizer_revision) is None:
        raise ValueError(f"model {model_id} tokenizer_revision must be a 40-hex commit")
    return ModelSpec(
        model_id=model_id,
        alias=_text(record.get("alias"), f"model {model_id} alias"),
        hf_repository=_text(
            record.get("hf_repository"), f"model {model_id} hf_repository"
        ),
        architecture=_text(record.get("architecture"), f"model {model_id} architecture"),
        parameterization=cast(Literal["dense", "mixture-of-experts"], parameterization),
        total_parameters_billion=total,
        active_parameters_billion=active,
        dtype=_text(record.get("dtype"), f"model {model_id} dtype"),
        quantization=_text(record.get("quantization"), f"model {model_id} quantization"),
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        base_url_env=raw_endpoint_env,
    )


def _optional_positive_integer(record: dict[str, Any], key: str, name: str) -> int | None:
    value = record.get(key)
    return None if value is None else _integer(value, f"{name}.{key}", minimum=1)


def _optional_positive_number(record: dict[str, Any], key: str, name: str) -> float | None:
    value = record.get(key)
    return None if value is None else _number(value, f"{name}.{key}", minimum=0.1)


def _load_stage(
    value: object,
    *,
    task_ids: set[str],
    arm_names: set[str],
    seeds: set[int],
    model_ids: set[str],
) -> StageSpec:
    record = _record(value, "stage")
    stage_id = _stable_id(record.get("id"), "stage.id")

    def text_selection(key: str, declared: set[str]) -> tuple[str, ...]:
        selected = tuple(
            _text(item, f"stage {stage_id} {key} item")
            for item in _sequence(record.get(key), f"stage {stage_id} {key}")
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"stage {stage_id} {key} must be nonempty and unique")
        unknown = set(selected) - declared
        if unknown:
            raise ValueError(f"stage {stage_id} has unknown {key}: {sorted(unknown)}")
        return selected

    selected_tasks = text_selection("tasks", task_ids)
    selected_arms = text_selection("arms", arm_names)
    selected_models = text_selection("models", model_ids)
    selected_seeds = tuple(
        _integer(item, f"stage {stage_id} seed", minimum=0)
        for item in _sequence(record.get("seeds"), f"stage {stage_id} seeds")
    )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError(f"stage {stage_id} seeds must be nonempty and unique")
    unknown_seeds = set(selected_seeds) - seeds
    if unknown_seeds:
        raise ValueError(f"stage {stage_id} has unknown seeds: {sorted(unknown_seeds)}")
    caps_record = _record(record.get("caps"), f"stage {stage_id} caps")
    allowed_caps = {
        "particles",
        "iterations",
        "max_scored_candidates",
        "wall_time_seconds",
        "provider_timeout_seconds",
    }
    unexpected_caps = set(caps_record) - allowed_caps
    if unexpected_caps:
        raise ValueError(
            f"stage {stage_id} has unsupported cap overrides: {sorted(unexpected_caps)}"
        )
    provider_budget = _integer(
        record.get("max_provider_scored_candidates"),
        f"stage {stage_id} max_provider_scored_candidates",
        minimum=0,
    )
    if any(arm in {"Q", "QD"} for arm in selected_arms) and provider_budget == 0:
        raise ValueError(f"provider-backed stage {stage_id} must have a positive budget")
    raw_requires_audit = record.get("requires_audit_stage")
    requires_audit_stage = (
        None
        if raw_requires_audit is None
        else _stable_id(raw_requires_audit, f"stage {stage_id} requires_audit_stage")
    )
    return StageSpec(
        stage_id=stage_id,
        description=_text(record.get("description"), f"stage {stage_id} description"),
        task_ids=selected_tasks,
        arms=tuple(cast(ArmName, arm) for arm in selected_arms),
        seeds=selected_seeds,
        model_ids=selected_models,
        caps=StageCaps(
            particles=_optional_positive_integer(caps_record, "particles", f"stage {stage_id}"),
            iterations=_optional_positive_integer(
                caps_record, "iterations", f"stage {stage_id}"
            ),
            max_scored_candidates=_optional_positive_integer(
                caps_record, "max_scored_candidates", f"stage {stage_id}"
            ),
            wall_time_seconds=_optional_positive_number(
                caps_record, "wall_time_seconds", f"stage {stage_id}"
            ),
            provider_timeout_seconds=_optional_positive_number(
                caps_record, "provider_timeout_seconds", f"stage {stage_id}"
            ),
        ),
        max_provider_scored_candidates=provider_budget,
        requires_audit_stage=requires_audit_stage,
    )


def effective_caps(protocol: Protocol, stage: StageSpec | None) -> Caps:
    """Apply a preregistered stage's narrow resource-cap overrides."""

    overrides = stage.caps if stage is not None else StageCaps()
    base = protocol.caps
    return Caps(
        particles=overrides.particles or base.particles,
        iterations=overrides.iterations or base.iterations,
        alpha=base.alpha,
        ess_threshold=base.ess_threshold,
        max_scored_candidates=(
            overrides.max_scored_candidates or base.max_scored_candidates
        ),
        support_limit=base.support_limit,
        hole_state_limit=base.hole_state_limit,
        hole_max_cost=base.hole_max_cost,
        wall_time_seconds=overrides.wall_time_seconds or base.wall_time_seconds,
        provider_timeout_seconds=(
            overrides.provider_timeout_seconds or base.provider_timeout_seconds
        ),
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
    schema_version = _integer(document.get("schema_version"), "schema_version", minimum=1)
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

    raw_models = document.get("models")
    provider_record = _record(document.get("provider"), "provider")
    legacy_model = provider_record.get("model")
    models: tuple[ModelSpec, ...]
    if raw_models is None and schema_version >= 2:
        raise ValueError("schema-v2 protocols must declare an ordered models array")
    if raw_models is None:
        # Keep schema-v1 protocols executable while making the lack of immutable
        # checkpoint metadata visible in every resulting manifest.
        legacy_alias = _text(legacy_model, "provider.model")
        models = (
            ModelSpec(
                model_id="legacy-unpinned",
                alias=legacy_alias,
                hf_repository="legacy-unpinned",
                architecture="legacy-unpinned",
                parameterization="dense",
                total_parameters_billion=1.0,
                active_parameters_billion=1.0,
                dtype="legacy-unpinned",
                quantization="legacy-unpinned",
                model_revision="legacy-unpinned",
                tokenizer_revision="legacy-unpinned",
                base_url_env=None,
            ),
        )
    else:
        models = tuple(_load_model(model) for model in _sequence(raw_models, "models"))
        if not models:
            raise ValueError("models must be a nonempty ordered array")
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("model ids must be unique")

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

    raw_api_env = provider_record.get("api_key_env")
    if raw_api_env is not None and not isinstance(raw_api_env, str):
        raise ValueError("provider.api_key_env must be null or a string")
    raw_server_config = provider_record.get("vllm_server_config")
    if raw_server_config is not None and not isinstance(raw_server_config, str):
        raise ValueError("provider.vllm_server_config must be null or a string")
    raw_cache_dir = provider_record.get("score_cache_dir")
    if raw_cache_dir is not None and not isinstance(raw_cache_dir, str):
        raise ValueError("provider.score_cache_dir must be null or a string")
    raw_cache_mode = provider_record.get("score_cache_mode", "off")
    if raw_cache_mode not in {"off", "read-write", "replay-only"}:
        raise ValueError("provider.score_cache_mode must be off, read-write, or replay-only")
    cache_dir = (
        None if raw_cache_dir is None else (resolved.parent / raw_cache_dir).resolve()
    )
    if raw_cache_mode != "off" and (cache_dir is None or not raw_server_config):
        raise ValueError(
            "non-off provider score_cache_mode requires score_cache_dir and "
            "vllm_server_config"
        )
    provider = ProviderSpec(
        model=legacy_model if isinstance(legacy_model, str) and legacy_model else None,
        base_url_env=_text(provider_record.get("base_url_env"), "provider.base_url_env"),
        api_key_env=raw_api_env,
        vllm_server_config=raw_server_config,
        score_cache_dir=cache_dir,
        score_cache_mode=cast(
            Literal["off", "read-write", "replay-only"], raw_cache_mode
        ),
    )
    shared = _record(document.get("shared_arguments"), "shared_arguments")
    allowed_shared = {
        "beta_max",
        "candidate_batch_size",
        "deduction_strength",
        "proposal_epsilon",
        "temperature",
        "llm_energy_normalization",
    }
    unexpected = set(shared) - allowed_shared
    if unexpected:
        raise ValueError(f"unsupported shared_arguments: {sorted(unexpected)}")
    if any(not isinstance(value, (str, int, float)) for value in shared.values()):
        raise ValueError("shared argument values must be scalar")
    normalization = shared.get("llm_energy_normalization")
    if normalization is None and schema_version == 1:
        # Historical schema-v1 runs inherited the core CLI's total-energy default.
        shared["llm_energy_normalization"] = "total-full-prompt-logprob"
    elif normalization is None:
        raise ValueError(
            "schema-v2 protocols must explicitly declare "
            "shared_arguments.llm_energy_normalization"
        )
    elif normalization != "mean-full-prompt-conditional-logprob":
        raise ValueError(
            "shared_arguments.llm_energy_normalization must be "
            "mean-full-prompt-conditional-logprob"
        )

    raw_materialize = document.get("materialize_reference", False)
    if not isinstance(raw_materialize, bool):
        raise ValueError("materialize_reference must be boolean")

    raw_target_contract = document.get("target_contract")
    target_contract: TargetContract | None = None
    if raw_target_contract is not None:
        if schema_version < 3:
            raise ValueError("target_contract requires schema_version 3 or newer")
        contract = _record(raw_target_contract, "target_contract")
        dominance = _text(
            contract.get("terminal_dominance_requirement"),
            "target_contract.terminal_dominance_requirement",
        )
        expected_dominance = (
            "min-exact-log-target-strictly-greater-than-max-inexact-log-target"
        )
        if dominance != expected_dominance:
            raise ValueError(
                "target_contract.terminal_dominance_requirement must be "
                f"{expected_dominance}"
            )
        required = contract.get("reference_audit_required")
        if not isinstance(required, bool):
            raise ValueError("target_contract.reference_audit_required must be boolean")
        target_contract = TargetContract(
            base_measure=cast(
                Literal["equal-family-within-family-occam"],
                _text(contract.get("base_measure"), "target_contract.base_measure"),
            ),
            likelihood=cast(
                Literal["soft-loss-gibbs"],
                _text(contract.get("likelihood"), "target_contract.likelihood"),
            ),
            loss_scale=_number(
                contract.get("loss_scale"), "target_contract.loss_scale", minimum=0.000001
            ),
            beta_max=_number(
                contract.get("beta_max"), "target_contract.beta_max", minimum=0.000001
            ),
            terminal_dominance_requirement=cast(
                Literal[
                    "min-exact-log-target-strictly-greater-than-max-inexact-log-target"
                ],
                dominance,
            ),
            reference_audit_required=required,
            reference_audit_stage=_stable_id(
                contract.get("reference_audit_stage"),
                "target_contract.reference_audit_stage",
            ),
        )
        if target_contract.base_measure != "equal-family-within-family-occam":
            raise ValueError(
                "target_contract.base_measure must be equal-family-within-family-occam"
            )
        if target_contract.likelihood != "soft-loss-gibbs":
            raise ValueError("target_contract.likelihood must be soft-loss-gibbs")

    task_ids = {task.task_id for task in tasks}
    arm_names = {str(arm.name) for arm in arms}
    seed_values = set(seeds)
    model_ids = {model.model_id for model in models}
    raw_stages = document.get("stages", [])
    stages = tuple(
        _load_stage(
            stage,
            task_ids=task_ids,
            arm_names=arm_names,
            seeds=seed_values,
            model_ids=model_ids,
        )
        for stage in _sequence(raw_stages, "stages")
    )
    if len({stage.stage_id for stage in stages}) != len(stages):
        raise ValueError("stage ids must be unique")
    stage_positions = {stage.stage_id: position for position, stage in enumerate(stages)}
    for position, stage in enumerate(stages):
        required_stage = stage.requires_audit_stage
        if required_stage is None:
            continue
        if required_stage not in stage_positions:
            raise ValueError(
                f"stage {stage.stage_id} requires unknown audit stage {required_stage}"
            )
        if stage_positions[required_stage] >= position:
            raise ValueError(
                f"stage {stage.stage_id} audit prerequisite must be declared earlier"
            )
    if target_contract is not None and target_contract.reference_audit_required:
        reference_id = target_contract.reference_audit_stage
        if reference_id not in stage_positions:
            raise ValueError(f"target contract references unknown audit stage {reference_id}")
        reference_stage = stages[stage_positions[reference_id]]
        if reference_stage.max_provider_scored_candidates != 0 or any(
            arm in {"Q", "QD"} for arm in reference_stage.arms
        ):
            raise ValueError("target-contract reference audit stage must be provider-free")
        for stage in stages:
            if not any(arm in {"Q", "QD"} for arm in stage.arms):
                continue
            if stage.requires_audit_stage != reference_id:
                raise ValueError(
                    f"provider-backed stage {stage.stage_id} must require audit stage "
                    f"{reference_id}"
                )
            uncovered_tasks = set(stage.task_ids) - set(reference_stage.task_ids)
            if uncovered_tasks:
                raise ValueError(
                    f"provider-backed stage {stage.stage_id} has tasks not covered by "
                    f"{reference_id}: {sorted(uncovered_tasks)}"
                )

    return Protocol(
        path=resolved,
        protocol_id=protocol_id,
        schema_version=schema_version,
        status=_text(document.get("status"), "status"),
        protocol_sha256=hashlib.sha256(content).hexdigest(),
        seeds=seeds,
        tasks=tasks,
        arms=arms,
        models=models,
        stages=stages,
        caps=caps,
        provider=provider,
        materialize_reference=raw_materialize,
        shared_arguments=cast(dict[str, str | int | float], shared),
        target_contract=target_contract,
    )

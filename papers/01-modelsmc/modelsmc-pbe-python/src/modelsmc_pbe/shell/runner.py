"""Application-level orchestration for one synthesis invocation."""

from __future__ import annotations

from pathlib import Path

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.runtime import DeviceInfo, resolve_device
from modelsmc_pbe.shell.execution import (
    CompletedRun,
    execute_grammar_smc,
    execute_importance_smc,
    execute_paper_search,
)
from modelsmc_pbe.shell.output import print_program_result
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.skeletons import resolve_skeleton


def _smc_overrides(request: SynthesizeRequest) -> dict[str, object]:
    overrides: dict[str, object] = {}
    optional_values = {
        "particles": request.particles,
        "iterations": request.iterations,
        "clone_probability": request.alpha,
        "seed": request.seed,
    }
    overrides.update(
        (name, value) for name, value in optional_values.items() if value is not None
    )
    if request.ess_threshold is not None:
        if request.ess_threshold <= 0:
            raise ValueError("ESS threshold must be greater than zero")
        overrides["ess_threshold"] = request.ess_threshold
    return overrides


def _runtime_overrides(request: SynthesizeRequest) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if request.device is not None:
        overrides["device"] = request.device
    if request.artifacts_dir is not None:
        overrides["artifacts_dir"] = request.artifacts_dir
    if request.trace:
        overrides["console_level"] = "trace"
    return overrides


def _load_config(request: SynthesizeRequest) -> ExperimentConfig:
    if request.mode not in {"paper-search", "grammar-smc", "importance-smc"}:
        raise ValueError("mode must be paper-search, grammar-smc, or importance-smc")
    return load_experiment_config(
        request.spec,
        smc_overrides=_smc_overrides(request),
        runtime_overrides=_runtime_overrides(request),
    )


def _selected_skeleton(
    request: SynthesizeRequest,
    config: ExperimentConfig,
) -> str | None:
    if request.mode == "grammar-smc" or (
        request.mode == "paper-search" and request.proposal == "catalog"
    ):
        return resolve_skeleton(config, request.skeleton)
    return None


def _create_logger(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    selected_skeleton: str | None,
) -> RunLogger:
    algorithm_config: dict[str, object] = {
        "experiment": config,
        "mode": request.mode,
        "proposal": request.proposal if request.mode != "grammar-smc" else None,
        "skeleton": selected_skeleton,
        "beta_max": (
            request.beta_max
            if request.mode in {"grammar-smc", "importance-smc"}
            else None
        ),
        "moves_per_stage": (
            request.moves_per_stage if request.mode == "grammar-smc" else None
        ),
        "grammar_limit": request.grammar_limit,
        "hole_max_cost": request.hole_max_cost,
        "hole_state_limit": request.hole_state_limit,
        "support_limit": request.support_limit,
        "proposal_epsilon": request.proposal_epsilon,
        "candidate_batch_size": request.candidate_batch_size,
        "model": (
            request.model
            if request.mode != "grammar-smc" and request.proposal != "catalog"
            else None
        ),
        "base_url": request.base_url,
    }
    claims = {
        "paper-search": "heuristic_search_uncorrected_proposal_kernel",
        "grammar-smc": "calibrated_finite_skeleton_conditioned_target",
        "importance-smc": "importance_corrected_finite_deduction_refuted_target",
    }
    claim = claims[request.mode]
    return RunLogger.create(
        base_dir=config.runtime.artifacts_dir,
        run_name=f"{config.spec.name}-{request.mode}",
        config=algorithm_config,
        device=device,
        seed=config.smc.seed,
        probabilistic_claim=claim,
        console_level=config.runtime.console_level,
        cwd=Path.cwd(),
    )


def _execute(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    logger: RunLogger,
) -> CompletedRun:
    if request.mode == "grammar-smc":
        return execute_grammar_smc(request, config, device, logger)
    if request.mode == "importance-smc":
        return execute_importance_smc(request, config, device, logger)
    return execute_paper_search(request, config, device, logger)


def run_synthesis(request: SynthesizeRequest) -> None:
    """Validate, execute, persist, and present one synthesis experiment."""

    config = _load_config(request)
    device = resolve_device(config.runtime.device)
    selected_skeleton = _selected_skeleton(request, config)
    logger = _create_logger(request, config, device, selected_skeleton)
    with logger:
        logger.event(
            "runtime.resolved",
            message="resolved experiment runtime",
            mode=request.mode,
            device=device.as_dict(),
            skeleton=selected_skeleton,
        )
        completed = _execute(request, config, device, logger)
        logger.finish(
            result=completed.persisted_result,
            final_particles=completed.final_particles,
        )
    print_program_result(completed.view, logger.run_dir)

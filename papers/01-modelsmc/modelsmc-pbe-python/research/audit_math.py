"""Audit probability, MAP, and telemetry math for the sealed deduction-stress matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer, RejectedProgram, ScoredProgram
from modelsmc_pbe.core.types import child
from modelsmc_pbe.domain import ProgramAst, canonical_key
from modelsmc_pbe.grammar import bounded_square_target
from modelsmc_pbe.induction import assemble_program
from modelsmc_pbe.proposals import LLMEnergyNormalization
from modelsmc_pbe.search.importance import (
    FactorizedSupportBuilder,
    ImportanceSMCOptions,
    replay_qwen_categorical,
)
from modelsmc_pbe.search.importance.factorized import (
    choice_guide_probabilities,
    family_guide_probabilities,
    trace_fillings,
    trace_log_prior,
    valid_choice_indices,
)
from modelsmc_pbe.search.importance.lazy_prompts import (
    lazy_family_prompt_prefix,
    lazy_hole_prompt_prefix,
)
from modelsmc_pbe.search.importance.lazy_records import (
    ConstructionTrace,
    FactorizedFamily,
    FactorizedImportanceSupport,
    LazyImportanceState,
)
from modelsmc_pbe.search.importance.prompts import (
    family_candidate,
    family_prompt_prefix,
    hole_prompt_prefix,
)
from modelsmc_pbe.search.importance.proposal_distribution import candidate_distribution
from modelsmc_pbe.search.importance.records import (
    FamilySupport,
    HoleFilling,
    ImportanceState,
    ImportanceSupport,
)
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder
from research.heldout import write_json_atomic

SCHEMA_VERSION = 1
IDENTIFIED = "IDENTIFIED"
NOT_IDENTIFIED = "NOT_IDENTIFIED"
NONIDENTIFYING_SPLICE_PROXY = "NONIDENTIFYING_SPLICE_PROXY"
REFUSED_NOT_IDENTIFIED = "REFUSED_NOT_IDENTIFIED"
DEFAULT_MATRIX = Path(__file__).with_name("outputs") / "deduction-stress-paired-v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _violation(
    violations: list[dict[str, str]],
    code: str,
    detail: str,
    *,
    cell_id: str | None = None,
) -> None:
    record = {"code": code, "detail": detail}
    if cell_id is not None:
        record["cell_id"] = cell_id
    violations.append(record)


def identified_probability(value: float, *, provenance: str) -> dict[str, object]:
    """Construct a probability record that may feed derived probability math."""

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("identified probability must be finite and in [0, 1]")
    return {"status": IDENTIFIED, "value": value, "provenance": provenance}


def probability_at_least_one(probability: Mapping[str, object], draws: int) -> float:
    """Return discovery probability, refusing unidentified or proxy inputs."""

    if probability.get("status") != IDENTIFIED:
        raise ValueError("derived discovery math requires an identified probability")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    value = _number(probability.get("value"), name="probability.value")
    return -math.expm1(draws * math.log1p(-value)) if value < 1.0 else 1.0


def draws_for_probability(
    probability: Mapping[str, object], target_probability: float = 0.5
) -> int:
    """Return the minimum integer draws for a target success probability."""

    if probability.get("status") != IDENTIFIED:
        raise ValueError("N-draw math requires an identified probability")
    value = _number(probability.get("value"), name="probability.value")
    if not 0.0 < value <= 1.0:
        raise ValueError("draw count is undefined for zero probability")
    if not math.isfinite(target_probability) or not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be in (0, 1)")
    if value == 1.0:
        return 1
    return math.ceil(math.log1p(-target_probability) / math.log1p(-value))


def _entropy(probabilities: Iterable[float]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)


def reconcile_telemetry(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Reconcile logical score work, provider work, and cached token positions."""

    violations: list[dict[str, str]] = []
    ledgers = _array(result.get("score_ledger"), name="result.score_ledger")
    ledger_records = tuple(
        _mapping(ledger, name="score ledger") for ledger in ledgers
    )
    ledger_candidates = 0
    ledger_token_positions = 0
    for record in ledger_records:
        candidates = _array(record.get("candidates"), name="score ledger candidates")
        ledger_candidates += len(candidates)
        ledger_token_positions += sum(
            _integer(
                _mapping(candidate, name="score candidate").get("scored_token_count"),
                name="scored_token_count",
            )
            for candidate in candidates
        )
    scored_candidates = _integer(result.get("scored_candidates"), name="scored_candidates")
    if ledger_candidates != scored_candidates:
        _violation(
            violations,
            "LEDGER_CANDIDATE_TOTAL_MISMATCH",
            f"ledger={ledger_candidates}, result={scored_candidates}",
            cell_id=cell_id,
        )

    metrics = _mapping(manifest.get("metrics", {}), name="manifest.metrics")
    cache = _mapping(metrics.get("candidate_score_cache", {}), name="cache metrics")
    provider = _mapping(metrics.get("candidate_score_provider", {}), name="provider metrics")

    def valid_digest(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    cache_mode = cache.get("mode", "off") if cache else "off"
    cache_ledgers: list[Mapping[str, Any]] = []
    provider_ledgers: list[Mapping[str, Any]] = []
    for index, ledger in enumerate(ledger_records):
        origin = ledger.get("score_origin")
        cache_hit = ledger.get("cache_hit")
        cache_key = ledger.get("cache_key_sha256")
        if cache_mode == "off":
            valid = origin in {"provider", "synthetic"} and cache_hit is None and cache_key is None
        elif origin == "cache":
            valid = cache_hit is True and valid_digest(cache_key)
            cache_ledgers.append(ledger)
        elif origin == "provider":
            valid = cache_hit is False and valid_digest(cache_key)
            provider_ledgers.append(ledger)
        else:
            valid = False
        if cache_mode == "off" and origin == "provider":
            provider_ledgers.append(ledger)
        if not valid:
            _violation(
                violations,
                "LEDGER_CACHE_PROVENANCE_INVALID",
                (
                    f"ledger={index}, mode={cache_mode!r}, origin={origin!r}, "
                    f"cache_hit={cache_hit!r}, cache_key={cache_key!r}"
                ),
                cell_id=cell_id,
            )

    if not cache or cache.get("mode") == "off":
        direct_provider_candidates = sum(
            len(_array(ledger.get("candidates"), name="score ledger candidates"))
            for ledger in provider_ledgers
        )
        direct_provider_tokens = sum(
            sum(
                _integer(
                    _mapping(candidate, name="score candidate").get(
                        "scored_token_count"
                    ),
                    name="scored_token_count",
                )
                for candidate in _array(
                    ledger.get("candidates"), name="score ledger candidates"
                )
            )
            for ledger in provider_ledgers
        )
        direct_provider_invocations = len(
            {(ledger.get("stage"), ledger.get("wave")) for ledger in provider_ledgers}
        )
        if provider:
            http_requests = _integer(
                provider.get("http_requests"), name="http_requests"
            )
            http_failures = _integer(
                provider.get("http_failures", 0), name="provider http_failures"
            )
            provider_tokens = _integer(
                provider.get("scored_token_positions"),
                name="provider scored_token_positions",
            )
            http_seconds = _number(
                provider.get("http_request_seconds_sum"),
                name="http_request_seconds_sum",
            )
        else:
            http_requests = 0
            http_failures = 0
            provider_tokens = 0
            http_seconds = 0.0
        if direct_provider_candidates and not provider:
            _violation(
                violations,
                "DIRECT_PROVIDER_METRICS_MISSING",
                f"provider_candidates={direct_provider_candidates}",
                cell_id=cell_id,
            )
        if provider_tokens != direct_provider_tokens:
            _violation(
                violations,
                "TOKEN_POSITION_RECONCILIATION_MISMATCH",
                (
                    f"provider={provider_tokens}, cache=0, "
                    f"provider_ledgers={direct_provider_tokens}"
                ),
                cell_id=cell_id,
            )
        if direct_provider_tokens != ledger_token_positions:
            _violation(
                violations,
                "CACHE_OFF_NONPROVIDER_TOKENS_PRESENT",
                (
                    f"provider_ledgers={direct_provider_tokens}, "
                    f"all_ledgers={ledger_token_positions}"
                ),
                cell_id=cell_id,
            )
        return (
            {
                "logical_score_requests": len(ledgers),
                "logical_candidates": scored_candidates,
                "provider_score_requests": len(provider_ledgers),
                "provider_invocations": direct_provider_invocations,
                "provider_invocation_failures": 0,
                "http_requests": http_requests,
                "provider_http_failures": http_failures,
                "provider_candidates": direct_provider_candidates,
                "cache_hit_candidates": 0,
                "provider_scored_token_positions": provider_tokens,
                "cache_served_token_positions": 0,
                "ledger_token_positions": ledger_token_positions,
                "http_request_seconds_sum": http_seconds,
                "provider_await_wall_seconds": None,
            },
            violations,
        )

    lookup_requests = _integer(cache.get("lookup_requests"), name="lookup_requests")
    hit_requests = _integer(cache.get("hit_requests"), name="hit_requests")
    miss_requests = _integer(cache.get("miss_requests"), name="miss_requests")
    provider_score_requests = _integer(
        cache.get("provider_score_requests"), name="provider_score_requests"
    )
    provider_invocations = _integer(
        cache.get("provider_invocations", 0), name="provider_invocations"
    )
    provider_failures = _integer(
        cache.get("provider_failures", 0), name="provider_failures"
    )
    lookup_candidates = _integer(cache.get("lookup_candidates"), name="lookup_candidates")
    hit_candidates = _integer(cache.get("hit_candidates"), name="hit_candidates")
    miss_candidates = _integer(cache.get("miss_candidates"), name="miss_candidates")
    provider_candidates = _integer(cache.get("provider_candidates"), name="provider_candidates")
    provider_tokens = _integer(
        cache.get("provider_scored_tokens"), name="provider_scored_tokens"
    )
    cache_tokens = _integer(
        cache.get("cache_served_scored_tokens"), name="cache_served_scored_tokens"
    )
    http_requests = _integer(provider.get("http_requests"), name="http_requests")
    http_failures = _integer(
        provider.get("http_failures", 0), name="provider http_failures"
    )
    event_provider_tokens = _integer(
        provider.get("scored_token_positions"), name="provider scored_token_positions"
    )
    cache_ledger_candidates = sum(
        len(_array(ledger.get("candidates"), name="score ledger candidates"))
        for ledger in cache_ledgers
    )
    provider_ledger_candidates = sum(
        len(_array(ledger.get("candidates"), name="score ledger candidates"))
        for ledger in provider_ledgers
    )
    cache_ledger_tokens = sum(
        sum(
            _integer(
                _mapping(candidate, name="score candidate").get("scored_token_count"),
                name="scored_token_count",
            )
            for candidate in _array(
                ledger.get("candidates"), name="score ledger candidates"
            )
        )
        for ledger in cache_ledgers
    )
    provider_ledger_tokens = sum(
        sum(
            _integer(
                _mapping(candidate, name="score candidate").get("scored_token_count"),
                name="scored_token_count",
            )
            for candidate in _array(
                ledger.get("candidates"), name="score ledger candidates"
            )
        )
        for ledger in provider_ledgers
    )

    equations = (
        (
            lookup_requests == len(ledgers),
            "LOGICAL_REQUEST_TOTAL_MISMATCH",
            f"lookup={lookup_requests}, ledgers={len(ledgers)}",
        ),
        (
            hit_requests + miss_requests == lookup_requests,
            "CACHE_REQUEST_PARTITION_MISMATCH",
            f"hits={hit_requests}, misses={miss_requests}, lookup={lookup_requests}",
        ),
        (
            provider_score_requests == miss_requests,
            "PROVIDER_LOGICAL_REQUEST_MISMATCH",
            f"provider={provider_score_requests}, misses={miss_requests}",
        ),
        (
            lookup_candidates == scored_candidates,
            "CACHE_CANDIDATE_TOTAL_MISMATCH",
            f"lookup={lookup_candidates}, scored={scored_candidates}",
        ),
        (
            hit_candidates + miss_candidates == lookup_candidates,
            "CACHE_CANDIDATE_PARTITION_MISMATCH",
            f"hits={hit_candidates}, misses={miss_candidates}, lookup={lookup_candidates}",
        ),
        (
            provider_candidates == miss_candidates,
            "PROVIDER_CANDIDATE_MISMATCH",
            f"provider={provider_candidates}, misses={miss_candidates}",
        ),
        (
            provider_tokens + cache_tokens == ledger_token_positions,
            "TOKEN_POSITION_RECONCILIATION_MISMATCH",
            (
                f"provider={provider_tokens}, cache={cache_tokens}, "
                f"ledger={ledger_token_positions}"
            ),
        ),
        (
            event_provider_tokens == provider_tokens,
            "PROVIDER_TOKEN_EVENT_MISMATCH",
            f"event={event_provider_tokens}, cache_summary={provider_tokens}",
        ),
        (
            len(cache_ledgers) == hit_requests
            and len(provider_ledgers) == miss_requests,
            "CACHE_LEDGER_REQUEST_RECONCILIATION_MISMATCH",
            (
                f"ledger_hits={len(cache_ledgers)}, metrics_hits={hit_requests}, "
                f"ledger_misses={len(provider_ledgers)}, metrics_misses={miss_requests}"
            ),
        ),
        (
            cache_ledger_candidates == hit_candidates
            and provider_ledger_candidates == miss_candidates,
            "CACHE_LEDGER_CANDIDATE_RECONCILIATION_MISMATCH",
            (
                f"ledger_hits={cache_ledger_candidates}, metrics_hits={hit_candidates}, "
                f"ledger_misses={provider_ledger_candidates}, metrics_misses={miss_candidates}"
            ),
        ),
        (
            cache_ledger_tokens == cache_tokens
            and provider_ledger_tokens == provider_tokens,
            "CACHE_LEDGER_TOKEN_RECONCILIATION_MISMATCH",
            (
                f"ledger_cache={cache_ledger_tokens}, metrics_cache={cache_tokens}, "
                f"ledger_provider={provider_ledger_tokens}, metrics_provider={provider_tokens}"
            ),
        ),
        (
            cache_mode != "replay-only" or not provider_ledgers,
            "REPLAY_ONLY_PROVIDER_LEDGER_PRESENT",
            f"provider_ledgers={len(provider_ledgers)}",
        ),
    )
    for valid, code, detail in equations:
        if not valid:
            _violation(violations, code, detail, cell_id=cell_id)

    return (
        {
            "logical_score_requests": lookup_requests,
            "logical_candidates": lookup_candidates,
            "provider_score_requests": provider_score_requests,
            "provider_invocations": provider_invocations,
            "provider_invocation_failures": provider_failures,
            "http_requests": http_requests,
            "provider_http_failures": http_failures,
            "provider_candidates": provider_candidates,
            "cache_hit_candidates": hit_candidates,
            "provider_scored_token_positions": provider_tokens,
            "cache_served_token_positions": cache_tokens,
            "ledger_token_positions": ledger_token_positions,
            "http_request_seconds_sum": _number(
                provider.get("http_request_seconds_sum"), name="http_request_seconds_sum"
            ),
            "provider_await_wall_seconds": _number(
                cache.get("provider_await_wall_seconds"),
                name="provider_await_wall_seconds",
            ),
        },
        violations,
    )


def _trace_records(family: FactorizedFamily) -> Iterable[ConstructionTrace]:
    ranges = (range(len(catalog.fillings)) for catalog in family.catalogs)
    for filling_indices in itertools.product(*ranges):
        cost = sum(
            catalog.costs[index]
            for catalog, index in zip(family.catalogs, filling_indices, strict=True)
        )
        if cost <= family.combination_budget:
            yield ConstructionTrace(
                hypothesis_index=family.hypothesis_index,
                filling_indices=tuple(filling_indices),
            )


def _state_summary(state: Mapping[str, Any]) -> dict[str, object]:
    return {
        "family": state["family"],
        "trace": state["trace"],
        "program": json.loads(str(state["key"])),
        "program_sha256": hashlib.sha256(str(state["key"]).encode()).hexdigest(),
        "loss": state["loss"],
        "cost": state["cost"],
        "log_prior": state["log_prior"],
        "log_target": state["log_target"],
    }


def _materialize_support_math(
    config: ExperimentConfig,
    options: ImportanceSMCOptions,
    target: ProgramAst,
) -> tuple[
    dict[str, object],
    FactorizedImportanceSupport,
    tuple[ConstructionTrace, ...],
    tuple[Mapping[str, Any], ...],
]:
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=options,
    ).build()
    scorer = ProgramScorer(config)
    target_key = canonical_key(target)
    all_states: list[dict[str, Any]] = []
    batch_traces: list[ConstructionTrace] = []
    batch_programs: list[ProgramAst] = []

    def flush() -> None:
        scores = scorer.score_batch(batch_programs)
        for trace, program, score in zip(batch_traces, batch_programs, scores, strict=True):
            if isinstance(score, RejectedProgram):
                raise RuntimeError(f"factorized support program was rejected: {score.reason}")
            assert isinstance(score, ScoredProgram)
            family = support.family(trace.hypothesis_index)
            log_prior = trace_log_prior(
                support,
                trace,
                cost_scale=float(config.smc.cost_scale),
            )
            key = canonical_key(program)
            all_states.append(
                {
                    "family": family.hypothesis.kind.value,
                    "trace": {
                        "hypothesis_index": trace.hypothesis_index,
                        "filling_indices": list(trace.filling_indices),
                    },
                    "trace_record": trace,
                    "key": key,
                    "loss": score.total_loss,
                    "cost": score.cost,
                    "exact": score.exact_program,
                    "log_prior": log_prior,
                    "log_target": log_prior
                    - options.beta_max
                    * float(config.smc.loss_scale)
                    * score.total_loss,
                }
            )
        batch_traces.clear()
        batch_programs.clear()

    for family in support.families:
        for trace in _trace_records(family):
            fillings = trace_fillings(family, trace)
            program = assemble_program(
                family.hypothesis,
                {filling.hole_name: filling.expression for filling in fillings},
                allowed_integer_constants=config.spec.integer_constants,
            )
            batch_traces.append(trace)
            batch_programs.append(program)
            if len(batch_programs) == options.score_batch_size:
                flush()
    if batch_programs:
        flush()

    exact = [state for state in all_states if state["exact"]]
    if not exact:
        raise RuntimeError("materialized support has no exact program")
    best_exact = max(exact, key=lambda state: (state["log_prior"], state["key"]))
    global_map = max(all_states, key=lambda state: (state["log_target"], state["key"]))
    nonexact = [state for state in all_states if not state["exact"]]
    best_nonexact = max(nonexact, key=lambda state: (state["log_target"], state["key"]))
    threshold_competitor = max(
        nonexact,
        key=lambda state: (
            (state["log_prior"] - best_exact["log_prior"]) / state["loss"],
            state["key"],
        ),
    )
    minimum_loss_scale = max(
        0.0,
        (threshold_competitor["log_prior"] - best_exact["log_prior"])
        / (options.beta_max * threshold_competitor["loss"]),
    )
    prior_total = math.fsum(math.exp(state["log_prior"]) for state in all_states)
    exact_prior_mass = math.fsum(math.exp(state["log_prior"]) for state in exact)
    maximum_log_target = max(float(state["log_target"]) for state in all_states)
    target_normalizer = math.fsum(
        math.exp(float(state["log_target"]) - maximum_log_target)
        for state in all_states
    )
    log_target_normalizer = maximum_log_target + math.log(target_normalizer)
    exact_target_mass = math.fsum(
        math.exp(float(state["log_target"]) - log_target_normalizer)
        for state in exact
    )
    target_matches = [state for state in exact if state["key"] == target_key]
    if not target_matches:
        raise RuntimeError("intended target is absent from the exact support")

    exact_traces = tuple(cast(ConstructionTrace, state["trace_record"]) for state in exact)
    report: dict[str, object] = {
        "support_states": support.support_states,
        "materialized_states": len(all_states),
        "exact_states": len(exact),
        "exact_support_fraction": len(exact) / support.support_states,
        "uniform_support_fraction_is_prior_mass": False,
        "prior_total": prior_total,
        "exact_prior_mass": exact_prior_mass,
        "exact_target_mass": exact_target_mass,
        "log_target_normalizer": log_target_normalizer,
        "intended_target_traces": len(target_matches),
        "configured_loss_scale": float(config.smc.loss_scale),
        "exact_is_global_map": bool(global_map["exact"]),
        "global_map": _state_summary(global_map),
        "best_exact": _state_summary(best_exact),
        "best_nonexact_at_configured_scale": _state_summary(best_nonexact),
        "minimum_loss_scale_for_exact_map_tie": minimum_loss_scale,
        "exact_is_unique_map_above_loss_scale": minimum_loss_scale,
        "threshold_competitor": _state_summary(threshold_competitor),
    }
    return report, support, exact_traces, tuple(all_states)


def _runtime_support(
    config: ExperimentConfig, options: ImportanceSMCOptions
) -> ImportanceSupport:
    """Build the exact indexed support used by the materialized runtime."""

    with ProgramScorer(config) as scorer:
        return ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()


def _d_guided_probability(
    config: ExperimentConfig,
    options: ImportanceSMCOptions,
    support: FactorizedImportanceSupport,
    exact_traces: Sequence[ConstructionTrace],
) -> dict[str, object]:
    violation_scale = options.deduction_strength
    family_guide = family_guide_probabilities(
        support,
        cost_scale=float(config.smc.cost_scale),
        violation_scale=violation_scale,
    )
    family_distribution = candidate_distribution(
        sequence_logprobs=tuple(0.0 for _ in support.families),
        q_deduction=family_guide,
        temperature=options.proposal_temperature,
        epsilon=options.proposal_epsilon,
        deduction_mix=options.resolved_family_deduction_mix,
    ).probabilities
    family_positions = {
        family.hypothesis_index: position for position, family in enumerate(support.families)
    }
    trace_probabilities: list[float] = []
    for trace in exact_traces:
        family = support.family(trace.hypothesis_index)
        probability = float(family_distribution[family_positions[trace.hypothesis_index]].item())
        prefix_cost = 0
        for hole_index, selected_index in enumerate(trace.filling_indices):
            choices = valid_choice_indices(
                family,
                hole_index=hole_index,
                prefix_cost=prefix_cost,
            )
            q_deduction = choice_guide_probabilities(
                family,
                hole_index=hole_index,
                prefix_cost=prefix_cost,
                choice_indices=choices,
                cost_scale=float(config.smc.cost_scale),
                violation_scale=violation_scale,
            )
            distribution = candidate_distribution(
                sequence_logprobs=tuple(0.0 for _ in choices),
                q_deduction=q_deduction,
                temperature=options.proposal_temperature,
                epsilon=options.proposal_epsilon,
                deduction_mix=options.resolved_hole_deduction_mix,
            ).probabilities
            position = choices.index(selected_index)
            probability *= float(distribution[position].item())
            prefix_cost += family.catalogs[hole_index].costs[selected_index]
        trace_probabilities.append(probability)
    probability_record = identified_probability(
        math.fsum(trace_probabilities),
        provenance="deterministic catalog-plus-deduction replay on exact target prefixes",
    )
    return {
        **probability_record,
        "family_deduction_mix": options.resolved_family_deduction_mix,
        "hole_deduction_mix": options.resolved_hole_deduction_mix,
        "exact_trace_probabilities": trace_probabilities,
        "probability_at_least_one_in_20_draws": probability_at_least_one(
            probability_record, 20
        ),
        "draws_for_50_percent": draws_for_probability(probability_record),
    }


def _candidate(
    ledger: Mapping[str, Any], canonical_candidate: str
) -> Mapping[str, Any] | None:
    for value in _array(ledger.get("candidates"), name="ledger candidates"):
        candidate = _mapping(value, name="ledger candidate")
        if candidate.get("canonical_candidate") == canonical_candidate:
            return candidate
    return None


def _previous_predicate(prompt_prefix: object) -> str | None:
    if not isinstance(prompt_prefix, str):
        return None
    marker = "PreviousFillings="
    for line in prompt_prefix.splitlines():
        if not line.startswith(marker):
            continue
        try:
            fillings = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(fillings, dict) or not isinstance(fillings.get("predicate"), dict):
            return None
        return canonical_key(cast(dict[str, Any], fillings["predicate"]))
    return None


def _replay_ledger(
    ledger: Mapping[str, Any], *, cell_id: str, violations: list[dict[str, str]]
) -> None:
    try:
        replayed = replay_qwen_categorical(ledger)
    except ValueError as error:
        _violation(
            violations,
            "QWEN_LEDGER_REPLAY_FAILED",
            str(error),
            cell_id=cell_id,
        )
        return
    candidates = _array(ledger.get("candidates"), name="ledger candidates")
    count = len(candidates)
    epsilon = _number(ledger.get("proposal_epsilon"), name="proposal_epsilon")
    deduction_mix = _number(ledger.get("deduction_mix"), name="deduction_mix")
    proposal: list[float] = []
    deduction_probabilities: list[float] = []
    canonical_candidates: list[str] = []
    for replayed_value, raw_candidate in zip(replayed, candidates, strict=True):
        candidate = _mapping(raw_candidate, name="ledger candidate")
        token_ids = _array(candidate.get("token_ids"), name="candidate token_ids")
        token_logprobs = _array(
            candidate.get("token_logprobs"), name="candidate token_logprobs"
        )
        if len(token_ids) != len(token_logprobs) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            _violation(
                violations,
                "CANDIDATE_TOKEN_ALIGNMENT_MISMATCH",
                f"token_ids={len(token_ids)}, token_logprobs={len(token_logprobs)}",
                cell_id=cell_id,
            )
        canonical_candidate = candidate.get("canonical_candidate")
        if not isinstance(canonical_candidate, str):
            _violation(
                violations,
                "CANDIDATE_CANONICAL_KEY_INVALID",
                "canonical_candidate must be a string",
                cell_id=cell_id,
            )
        else:
            canonical_candidates.append(canonical_candidate)
        deduction = _number(
            candidate.get("deduction_probability"), name="deduction_probability"
        )
        deduction_probabilities.append(deduction)
        expected = (1.0 - epsilon) * (
            (1.0 - deduction_mix) * replayed_value + deduction_mix * deduction
        ) + epsilon / count
        stored = _number(candidate.get("proposal_probability"), name="proposal_probability")
        if not math.isclose(expected, stored, rel_tol=1e-12, abs_tol=1e-12):
            _violation(
                violations,
                "PROPOSAL_MIXTURE_REPLAY_FAILED",
                f"expected={expected}, stored={stored}",
                cell_id=cell_id,
            )
        proposal.append(stored)
    if len(canonical_candidates) != len(set(canonical_candidates)):
        _violation(
            violations,
            "CANDIDATE_CANONICAL_KEY_DUPLICATE",
            "candidate catalog contains duplicate canonical keys",
            cell_id=cell_id,
        )
    if any(value < 0.0 for value in deduction_probabilities) or not math.isclose(
        math.fsum(deduction_probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        _violation(
            violations,
            "DEDUCTION_DISTRIBUTION_INVALID",
            f"sum={math.fsum(deduction_probabilities)}",
            cell_id=cell_id,
        )
    if not math.isclose(math.fsum(proposal), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        _violation(
            violations,
            "PROPOSAL_NOT_NORMALIZED",
            f"sum={math.fsum(proposal)}",
            cell_id=cell_id,
        )
    for raw_selection in _array(ledger.get("selections"), name="ledger selections"):
        selection = _mapping(raw_selection, name="ledger selection")
        selected_index = _integer(selection.get("selected_index"), name="selected_index")
        if not 0 <= selected_index < count:
            _violation(
                violations,
                "SELECTION_INDEX_OUT_OF_RANGE",
                f"selected_index={selected_index}, candidates={count}",
                cell_id=cell_id,
            )
            continue
        selected = _mapping(candidates[selected_index], name="selected candidate")
        selected_equations = (
            (
                "selected_probability",
                _number(selected.get("proposal_probability"), name="proposal_probability"),
            ),
            (
                "selected_qwen_probability",
                _number(selected.get("qwen_probability"), name="qwen_probability"),
            ),
            (
                "selected_deduction_probability",
                _number(
                    selected.get("deduction_probability"),
                    name="deduction_probability",
                ),
            ),
        )
        for field, expected in selected_equations:
            stored = _number(selection.get(field), name=field)
            if not math.isclose(expected, stored, rel_tol=1e-12, abs_tol=1e-12):
                _violation(
                    violations,
                    "SELECTION_PROBABILITY_MISMATCH",
                    f"{field}: expected={expected}, stored={stored}",
                    cell_id=cell_id,
                )


def audit_qd_target_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    exact_predicate_keys: frozenset[str],
    mapper_key: str,
    cell_id: str,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Replay QD ledgers and keep proxy products separate from identified paths."""

    violations: list[dict[str, str]] = []
    family_by_slot: dict[int, float] = {}
    predicate_by_slot: dict[int, float] = {}
    mapper_by_slot: dict[int, float] = {}
    exact_prefix_mapper_requests = 0
    mapper_qwen_ranks: list[int] = []
    mapper_proposal_ranks: list[int] = []
    predicate_qwen_ranks: list[int] = []
    predicate_proposal_ranks: list[int] = []
    normalized_entropies: list[float] = []

    for ledger in ledgers:
        _replay_ledger(ledger, cell_id=cell_id, violations=violations)
        candidates = [
            _mapping(value, name="ledger candidate")
            for value in _array(ledger.get("candidates"), name="ledger candidates")
        ]
        probabilities = [
            _number(candidate.get("qwen_probability"), name="qwen_probability")
            for candidate in candidates
        ]
        if len(probabilities) > 1:
            normalized_entropies.append(_entropy(probabilities) / math.log(len(probabilities)))
        selections = [
            _mapping(value, name="ledger selection")
            for value in _array(ledger.get("selections"), name="ledger selections")
        ]
        family = _candidate(ledger, '"foldr-filter-map"')
        if ledger.get("wave") == "family" and family is not None:
            probability = _number(
                family.get("proposal_probability"), name="family proposal_probability"
            )
            for selection in selections:
                family_by_slot[_integer(selection.get("slot"), name="slot")] = probability

        exact_predicates = [
            candidate
            for candidate in candidates
            if candidate.get("canonical_candidate") in exact_predicate_keys
        ]
        if len(exact_predicates) == len(exact_predicate_keys):
            probability = math.fsum(
                _number(candidate.get("proposal_probability"), name="predicate probability")
                for candidate in exact_predicates
            )
            for exact_predicate in exact_predicates:
                qwen_probability = _number(
                    exact_predicate.get("qwen_probability"),
                    name="predicate qwen_probability",
                )
                proposal_probability = _number(
                    exact_predicate.get("proposal_probability"),
                    name="predicate proposal_probability",
                )
                predicate_qwen_ranks.append(
                    1
                    + sum(
                        _number(candidate.get("qwen_probability"), name="qwen_probability")
                        > qwen_probability
                        for candidate in candidates
                    )
                )
                predicate_proposal_ranks.append(
                    1
                    + sum(
                        _number(
                            candidate.get("proposal_probability"),
                            name="proposal_probability",
                        )
                        > proposal_probability
                        for candidate in candidates
                    )
                )
            for selection in selections:
                predicate_by_slot[_integer(selection.get("slot"), name="slot")] = probability

        mapper = _candidate(ledger, mapper_key)
        if mapper is None:
            continue
        previous_predicate = _previous_predicate(ledger.get("prompt_prefix"))
        if previous_predicate in exact_predicate_keys:
            exact_prefix_mapper_requests += 1
        probability = _number(
            mapper.get("proposal_probability"), name="mapper proposal_probability"
        )
        qwen_probability = _number(
            mapper.get("qwen_probability"), name="mapper qwen_probability"
        )
        mapper_qwen_ranks.append(
            1
            + sum(
                _number(candidate.get("qwen_probability"), name="qwen_probability")
                > qwen_probability
                for candidate in candidates
            )
        )
        mapper_proposal_ranks.append(
            1
            + sum(
                _number(candidate.get("proposal_probability"), name="proposal_probability")
                > probability
                for candidate in candidates
            )
        )
        for selection in selections:
            mapper_by_slot[_integer(selection.get("slot"), name="slot")] = probability

    splice_values = [
        family_by_slot[slot] * predicate_by_slot[slot] * mapper_by_slot[slot]
        for slot in sorted(family_by_slot.keys() & predicate_by_slot.keys() & mapper_by_slot.keys())
    ]
    if exact_prefix_mapper_requests == 0:
        exact_proposal: dict[str, object] = {
            "status": NOT_IDENTIFIED,
            "value": None,
            "reason": (
                "no mapper score request is conditioned on either exact predicate prefix; "
                "sibling-prefix mapper scores cannot identify the exact path"
            ),
            "missing_component": "exact-prefix mapped_value request",
        }
    else:
        exact_proposal = {
            "status": NOT_IDENTIFIED,
            "value": None,
            "reason": "full target-prefix chain replay is required",
        }
    splice: dict[str, object] = {
        "status": NONIDENTIFYING_SPLICE_PROXY,
        "eligible_for_discovery_math": False,
        "observations": len(splice_values),
        "mean": None if not splice_values else math.fsum(splice_values) / len(splice_values),
        "minimum": None if not splice_values else min(splice_values),
        "maximum": None if not splice_values else max(splice_values),
        "reason": "mapper factors come from sampled non-target predicate prefixes",
    }
    try:
        probability_at_least_one(splice, 20)
    except ValueError as error:
        discovery: dict[str, object] = {
            "status": REFUSED_NOT_IDENTIFIED,
            "value": None,
            "reason": str(error),
        }
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("splice proxy unexpectedly entered discovery math")
    try:
        draws_for_probability(splice)
    except ValueError as error:
        n50: dict[str, object] = {
            "status": REFUSED_NOT_IDENTIFIED,
            "value": None,
            "reason": str(error),
        }
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("splice proxy unexpectedly entered N50 math")

    return (
        {
            "exact_proposal_probability": exact_proposal,
            "exact_prefix_mapper_requests": exact_prefix_mapper_requests,
            "splice_proxy": splice,
            "discovery_probability_20_draws": discovery,
            "draws_for_50_percent": n50,
            "mapper_qwen_ranks": mapper_qwen_ranks,
            "mapper_proposal_ranks": mapper_proposal_ranks,
            "predicate_qwen_ranks": predicate_qwen_ranks,
            "predicate_proposal_ranks": predicate_proposal_ranks,
            "normalized_qwen_entropy_range": (
                None
                if not normalized_entropies
                else [min(normalized_entropies), max(normalized_entropies)]
            ),
        },
        violations,
    )


def _target_hole_keys() -> tuple[frozenset[str], str]:
    target = bounded_square_target()
    reducer = child(target, "reducer")
    predicate = cast(dict[str, Any], child(reducer, "condition"))
    mapped = cast(dict[str, Any], child(child(reducer, "thenExpr"), "head"))
    swapped = {
        "kind": "And",
        "left": predicate["right"],
        "right": predicate["left"],
    }
    return frozenset((canonical_key(predicate), canonical_key(swapped))), canonical_key(mapped)


def _arm_document(protocol: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    for raw_arm in _array(protocol.get("arms"), name="protocol.arms"):
        arm = _mapping(raw_arm, name="protocol arm")
        if arm.get("id") == arm_id:
            return arm
    raise ValueError(f"deduction-stress protocol lacks arm {arm_id!r}")


def _resolved_arm_mixes(
    protocol: Mapping[str, Any], arm_id: str
) -> tuple[float, float, float]:
    arm = _arm_document(protocol, arm_id)
    fallback = _number(arm.get("deduction_mix"), name=f"{arm_id}.deduction_mix")
    family_value = arm.get("family_deduction_mix")
    hole_value = arm.get("hole_deduction_mix")
    family = (
        fallback
        if family_value is None
        else _number(family_value, name=f"{arm_id}.family_deduction_mix")
    )
    hole = (
        fallback
        if hole_value is None
        else _number(hole_value, name=f"{arm_id}.hole_deduction_mix")
    )
    return fallback, family, hole


def _options(protocol: Mapping[str, Any]) -> ImportanceSMCOptions:
    caps = _mapping(protocol.get("caps"), name="protocol.caps")
    shared = _mapping(protocol.get("shared_arguments"), name="protocol.shared_arguments")
    deduction_mix, family_deduction_mix, hole_deduction_mix = _resolved_arm_mixes(
        protocol, "D"
    )
    raw_normalization = shared.get("llm_energy_normalization")
    if not isinstance(raw_normalization, str):
        raise ValueError("llm_energy_normalization must be a string")
    return ImportanceSMCOptions(
        hole_max_cost=_integer(caps.get("hole_max_cost"), name="hole_max_cost"),
        hole_state_limit=_integer(caps.get("hole_state_limit"), name="hole_state_limit"),
        support_limit=_integer(caps.get("support_limit"), name="support_limit"),
        proposal_temperature=_number(shared.get("temperature"), name="temperature"),
        proposal_epsilon=_number(shared.get("proposal_epsilon"), name="proposal_epsilon"),
        deduction_mix=deduction_mix,
        family_deduction_mix=family_deduction_mix,
        hole_deduction_mix=hole_deduction_mix,
        deduction_strength=_number(
            shared.get("deduction_strength"), name="deduction_strength"
        ),
        beta_max=_number(shared.get("beta_max"), name="beta_max"),
        max_scored_candidates=_integer(
            caps.get("max_scored_candidates"), name="max_scored_candidates"
        ),
        multi_family=True,
        llm_energy_normalization=LLMEnergyNormalization(raw_normalization),
    )


def _experiment_path(protocol: Mapping[str, Any]) -> Path:
    tasks = _array(protocol.get("tasks"), name="protocol.tasks")
    if len(tasks) != 1:
        raise ValueError("deduction-stress audit expects exactly one task")
    task = _mapping(tasks[0], name="protocol task")
    if task.get("id") != "foldr-sparse-bounded-square":
        raise ValueError("audit_math is scoped to foldr-sparse-bounded-square")
    spec = task.get("spec")
    if not isinstance(spec, str):
        raise ValueError("protocol task spec must be a string")
    return Path(__file__).parents[1] / "examples" / Path(spec).name


def _selection_signature(ledgers: Sequence[Mapping[str, Any]]) -> list[list[object]]:
    signature: list[list[object]] = []
    for ledger in ledgers:
        for raw_selection in _array(ledger.get("selections"), name="ledger selections"):
            selection = _mapping(raw_selection, name="ledger selection")
            signature.append(
                [
                    ledger.get("wave"),
                    ledger.get("request_index"),
                    ledger.get("prompt_prefix_sha256"),
                    selection.get("slot"),
                    selection.get("ancestor_state_index"),
                    selection.get("selected_index"),
                ]
            )
    return signature


def _audit_matrix_protocol_stage(
    matrix_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    planned: Mapping[str, Mapping[str, Any]],
    *,
    violations: list[dict[str, str]],
) -> None:
    """Rebuild stage caps and the expected task/arm/seed/model population."""

    stage_id = matrix_manifest.get("stage_id")
    stages = [
        _mapping(raw_stage, name="protocol stage")
        for raw_stage in _array(protocol.get("stages"), name="protocol.stages")
        if _mapping(raw_stage, name="protocol stage").get("id") == stage_id
    ]
    if len(stages) != 1:
        _violation(
            violations,
            "PROTOCOL_STAGE_ID_MISMATCH",
            f"stage_id={stage_id!r}, matches={len(stages)}",
        )
        return
    stage = stages[0]
    expected_caps = dict(_mapping(protocol.get("caps"), name="protocol.caps"))
    expected_caps.update(_mapping(stage.get("caps", {}), name="stage.caps"))
    observed_caps = _mapping(
        matrix_manifest.get("effective_caps"), name="effective_caps"
    )
    if expected_caps != dict(observed_caps):
        _violation(
            violations,
            "EFFECTIVE_CAPS_MISMATCH",
            f"expected={expected_caps}, observed={dict(observed_caps)}",
        )

    arm_documents = {
        str(arm.get("id")): arm
        for arm in (
            _mapping(raw_arm, name="protocol arm")
            for raw_arm in _array(protocol.get("arms"), name="protocol.arms")
        )
    }
    model_documents = {
        str(model.get("id")): model
        for model in (
            _mapping(raw_model, name="protocol model")
            for raw_model in _array(protocol.get("models"), name="protocol.models")
        )
    }
    task_documents = {
        str(task.get("id")): task
        for task in (
            _mapping(raw_task, name="protocol task")
            for raw_task in _array(protocol.get("tasks"), name="protocol.tasks")
        )
    }
    expected_population: set[tuple[str, str, int, str | None]] = set()
    for task_id in _array(stage.get("tasks"), name="stage.tasks"):
        if not isinstance(task_id, str):
            raise ValueError("stage task id must be a string")
        for arm_id in _array(stage.get("arms"), name="stage.arms"):
            if not isinstance(arm_id, str) or arm_id not in arm_documents:
                raise ValueError("stage arm id is invalid")
            arm = arm_documents[arm_id]
            for seed in _array(stage.get("seeds"), name="stage.seeds"):
                parsed_seed = _integer(seed, name="stage seed")
                if arm.get("proposal") == "catalog":
                    expected_population.add((task_id, arm_id, parsed_seed, None))
                else:
                    for model_id in _array(stage.get("models"), name="stage.models"):
                        if not isinstance(model_id, str):
                            raise ValueError("stage model id must be a string")
                        expected_population.add(
                            (task_id, arm_id, parsed_seed, model_id)
                        )
    observed_population = {
        (
            str(cell.get("task_id")),
            str(cell.get("arm")),
            _integer(cell.get("seed"), name="planned cell seed"),
            cast(str | None, cell.get("model_id")),
        )
        for cell in planned.values()
    }
    model_field_mapping = {
        "model_alias": "alias",
        "model_hf_repository": "hf_repository",
        "model_architecture": "architecture",
        "model_parameterization": "parameterization",
        "model_total_parameters_billion": "total_parameters_billion",
        "model_active_parameters_billion": "active_parameters_billion",
        "model_dtype": "dtype",
        "model_quantization": "quantization",
        "model_revision": "model_revision",
        "tokenizer_revision": "tokenizer_revision",
    }
    for cell_id, cell in planned.items():
        task = task_documents.get(str(cell.get("task_id")))
        if task is None or cell.get("analysis_label") != task.get("analysis_label"):
            _violation(
                violations,
                "PLANNED_TASK_METADATA_MISMATCH",
                cell_id,
            )
        model_id = cell.get("model_id")
        if model_id is None:
            if any(cell.get(field) is not None for field in model_field_mapping):
                _violation(
                    violations,
                    "SIZE_INVARIANT_CELL_HAS_MODEL_METADATA",
                    cell_id,
                )
            continue
        model = model_documents.get(str(model_id))
        if model is None or any(
            cell.get(cell_field) != model.get(protocol_field)
            for cell_field, protocol_field in model_field_mapping.items()
        ):
            _violation(
                violations,
                "PLANNED_MODEL_METADATA_MISMATCH",
                cell_id,
            )
    if expected_population != observed_population:
        _violation(
            violations,
            "STAGE_POPULATION_MISMATCH",
            (
                "missing="
                f"{sorted(expected_population - observed_population, key=repr)}, "
                "extra="
                f"{sorted(observed_population - expected_population, key=repr)}"
            ),
        )


def _particle_path_signature(particle: Mapping[str, Any]) -> list[object]:
    """Return the persisted path identity for old trace and current state schemas."""

    has_trace_fields = "trace" in particle or "ancestor_trace" in particle
    has_state_fields = "state_index" in particle or "ancestor_state_index" in particle
    if has_trace_fields and has_state_fields:
        raise ValueError("particle mixes trace and state-index path identities")
    trace = particle.get("trace")
    ancestor_trace = particle.get("ancestor_trace")
    if has_trace_fields:
        if not isinstance(trace, dict) or not isinstance(ancestor_trace, dict):
            raise ValueError("particle trace identity is incomplete")
        return ["trace", trace, ancestor_trace]
    if not has_state_fields:
        raise ValueError("particle lacks a persisted path identity")
    state_index = particle.get("state_index")
    ancestor_state_index = particle.get("ancestor_state_index")
    return [
        "state_index",
        _integer(state_index, name="particle state_index"),
        _integer(ancestor_state_index, name="particle ancestor_state_index"),
    ]


def _result_search_summary(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int | None, str]:
    """Read search evidence without conflating historical and finite schemas."""

    if "search" in result:
        raw_search = result.get("search")
        if not isinstance(raw_search, dict):
            raise ValueError("result.search must be an object when present")
        best = _mapping(result.get("best_visited"), name="result.best_visited")
        evaluated = _integer(
            raw_search.get("evaluated_programs"), name="search.evaluated_programs"
        )
        return best, evaluated, "best_visited"
    if "best_visited" in result:
        raise ValueError("result.best_visited requires result.search")
    best = _mapping(result.get("sampled_best"), name="result.sampled_best")
    return best, None, "sampled_best_final_population"


def _audit_result_mixes(
    result: Mapping[str, Any],
    ledgers: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    arm: str,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Check protocol, persisted result, and wave-local mixture agreement."""

    expected_fallback, expected_family, expected_hole = _resolved_arm_mixes(protocol, arm)
    stored_fallback = _number(result.get("deduction_mix"), name="result.deduction_mix")
    raw_family = result.get("family_deduction_mix")
    raw_hole = result.get("hole_deduction_mix")
    stored_family = (
        stored_fallback
        if raw_family is None
        else _number(raw_family, name="result.family_deduction_mix")
    )
    stored_hole = (
        stored_fallback
        if raw_hole is None
        else _number(raw_hole, name="result.hole_deduction_mix")
    )
    for name, expected, stored in (
        ("deduction_mix", expected_fallback, stored_fallback),
        ("family_deduction_mix", expected_family, stored_family),
        ("hole_deduction_mix", expected_hole, stored_hole),
    ):
        if not math.isclose(expected, stored, rel_tol=0.0, abs_tol=1e-15):
            _violation(
                violations,
                "RESULT_MIX_CONFIG_MISMATCH",
                f"{name}: protocol={expected}, result={stored}",
                cell_id=cell_id,
            )
    for ledger in ledgers:
        wave = ledger.get("wave")
        if wave == "family":
            expected = expected_family
        elif isinstance(wave, str) and wave.startswith("hole-"):
            expected = expected_hole
        else:
            _violation(
                violations,
                "LEDGER_WAVE_INVALID",
                f"wave={wave!r}",
                cell_id=cell_id,
            )
            continue
        stored = _number(ledger.get("deduction_mix"), name="ledger.deduction_mix")
        if not math.isclose(expected, stored, rel_tol=0.0, abs_tol=1e-15):
            _violation(
                violations,
                "LEDGER_WAVE_MIX_MISMATCH",
                f"wave={wave}, protocol={expected}, ledger={stored}",
                cell_id=cell_id,
            )


def _audit_protocol_configuration(
    result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    cell_document: Mapping[str, Any],
    core_manifest: Mapping[str, Any],
    matrix_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cell_plan: Mapping[str, Any],
    ledgers: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Bind runtime/result/ledger settings to the frozen protocol and cell plan."""

    shared = _mapping(protocol.get("shared_arguments"), name="protocol.shared_arguments")
    caps = _mapping(matrix_manifest.get("effective_caps"), name="effective_caps")
    arm_id = str(cell_plan.get("arm"))
    arm = _arm_document(protocol, arm_id)
    provider_document = _mapping(protocol.get("provider"), name="protocol.provider")
    runtime = _mapping(core_manifest.get("configuration"), name="configuration")
    metrics = _mapping(core_manifest.get("metrics", {}), name="manifest.metrics")
    cache_metrics = _mapping(
        metrics.get("candidate_score_cache", {}), name="candidate_score_cache"
    )
    experiment = _mapping(runtime.get("experiment"), name="configuration.experiment")
    smc = _mapping(experiment.get("smc"), name="configuration.experiment.smc")
    embedded_spec = _mapping(
        experiment.get("spec"), name="configuration.experiment.spec"
    )
    fallback_mix, family_mix, hole_mix = _resolved_arm_mixes(protocol, arm_id)

    mismatches: list[str] = []

    def check(name: str, stored: object, expected: object) -> None:
        if isinstance(expected, float):
            if isinstance(stored, bool) or not isinstance(stored, (int, float)):
                mismatches.append(f"{name}={stored!r}, expected={expected!r}")
                return
            if not math.isclose(float(stored), expected, rel_tol=0.0, abs_tol=1e-15):
                mismatches.append(f"{name}={stored!r}, expected={expected!r}")
        elif stored != expected:
            mismatches.append(f"{name}={stored!r}, expected={expected!r}")

    def path_identity(value: object) -> tuple[str, ...] | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        return tuple(
            part for part in path.parts if part not in {path.anchor, ".", ".."}
        )

    def check_path_identity(name: str, stored: object, expected: object) -> None:
        expected_parts = path_identity(expected)
        if expected_parts is None:
            check(name, stored, None)
            return
        stored_parts = path_identity(stored)
        if (
            stored_parts is None
            or len(stored_parts) < len(expected_parts)
            or stored_parts[-len(expected_parts) :] != expected_parts
        ):
            mismatches.append(
                f"{name}={stored!r}, expected_suffix={expected_parts!r}"
            )

    expected_temperature = _number(shared.get("temperature"), name="temperature")
    expected_epsilon = _number(
        shared.get("proposal_epsilon"), name="proposal_epsilon"
    )
    expected_beta = _number(shared.get("beta_max"), name="beta_max")
    expected_strength = _number(
        shared.get("deduction_strength"), name="deduction_strength"
    )
    expected_normalization = shared.get("llm_energy_normalization")
    expected_particles = _integer(caps.get("particles"), name="particles")
    expected_iterations = _integer(caps.get("iterations"), name="iterations")
    expected_alpha = _number(caps.get("alpha"), name="alpha")
    expected_ess = _number(caps.get("ess_threshold"), name="ess_threshold")
    expected_score_cap = _integer(
        caps.get("max_scored_candidates"), name="max_scored_candidates"
    )
    expected_timeout = _number(
        caps.get("provider_timeout_seconds"), name="provider_timeout_seconds"
    )
    expected_materialize = protocol.get("materialize_reference")
    if not isinstance(expected_materialize, bool):
        raise ValueError("protocol.materialize_reference must be a boolean")
    provider_backed = arm.get("proposal") == "vllm"
    expected_cache_mode = (
        provider_document.get("score_cache_mode", "off") if provider_backed else "off"
    )
    if expected_cache_mode not in {"off", "read-write", "replay-only"}:
        raise ValueError("protocol provider cache mode is invalid")
    expected_cache_dir = (
        provider_document.get("score_cache_dir")
        if provider_backed and expected_cache_mode != "off"
        else None
    )
    expected_server_config = (
        provider_document.get("vllm_server_config")
        if provider_backed and expected_cache_mode != "off"
        else None
    )
    expected_spec = config.spec.model_dump(mode="json", by_alias=True)
    expected_smc = config.smc.model_dump(mode="json", by_alias=True)
    expected_smc.update(
        {
            "particles": expected_particles,
            "iterations": expected_iterations,
            "cloneProbability": expected_alpha,
            "essThreshold": expected_ess,
            "seed": cell_plan.get("seed"),
        }
    )
    expected_proposal_source = (
        "uniform-finite-candidates"
        if arm.get("proposal") == "catalog"
        else "vllm-prompt-logprobs"
    )

    check("envelope.particle_count", envelope.get("particle_count"), expected_particles)
    check("manifest.seed", core_manifest.get("seed"), cell_plan.get("seed"))
    check("runtime.beta_max", runtime.get("beta_max"), expected_beta)
    check(
        "runtime.candidate_batch_size",
        runtime.get("candidate_batch_size"),
        shared.get("candidate_batch_size"),
    )
    check("runtime.deduction_mix", runtime.get("deduction_mix"), fallback_mix)
    check(
        "runtime.family_deduction_mix",
        runtime.get("family_deduction_mix", runtime.get("deduction_mix")),
        family_mix,
    )
    check(
        "runtime.hole_deduction_mix",
        runtime.get("hole_deduction_mix", runtime.get("deduction_mix")),
        hole_mix,
    )
    check("runtime.deduction_strength", runtime.get("deduction_strength"), expected_strength)
    check("runtime.proposal_epsilon", runtime.get("proposal_epsilon"), expected_epsilon)
    check("runtime.temperature", runtime.get("temperature"), expected_temperature)
    check(
        "runtime.llm_energy_normalization",
        runtime.get("llm_energy_normalization"),
        expected_normalization,
    )
    check("runtime.max_scored_candidates", runtime.get("max_scored_candidates"), expected_score_cap)
    check("runtime.hole_max_cost", runtime.get("hole_max_cost"), caps.get("hole_max_cost"))
    check(
        "runtime.hole_state_limit",
        runtime.get("hole_state_limit"),
        caps.get("hole_state_limit"),
    )
    check("runtime.support_limit", runtime.get("support_limit"), caps.get("support_limit"))
    check("runtime.proposal", runtime.get("proposal"), arm.get("proposal"))
    check("runtime.mode", runtime.get("mode"), "importance-smc")
    check("runtime.timeout_seconds", runtime.get("timeout_seconds"), expected_timeout)
    check(
        "runtime.materialize_reference",
        runtime.get("materialize_reference"),
        expected_materialize,
    )
    check("runtime.requested_skeleton", runtime.get("requested_skeleton"), "auto")
    check("runtime.skeleton", runtime.get("skeleton"), "multi-family")
    check("runtime.score_cache_mode", runtime.get("score_cache_mode"), expected_cache_mode)
    check_path_identity(
        "runtime.score_cache_dir", runtime.get("score_cache_dir"), expected_cache_dir
    )
    check("metrics.score_cache_mode", cache_metrics.get("mode"), expected_cache_mode)
    check_path_identity(
        "metrics.score_cache_dir", cache_metrics.get("cache_dir"), expected_cache_dir
    )
    check(
        "runtime.vllm_server_config",
        runtime.get("vllm_server_config"),
        expected_server_config,
    )
    check("runtime.experiment.spec", dict(embedded_spec), expected_spec)
    check("runtime.experiment.smc", dict(smc), expected_smc)
    check("smc.cloneProbability", smc.get("cloneProbability"), expected_alpha)
    check("smc.essThreshold", smc.get("essThreshold"), expected_ess)
    check("smc.iterations", smc.get("iterations"), expected_iterations)
    check("smc.particles", smc.get("particles"), expected_particles)
    check("smc.seed", smc.get("seed"), cell_plan.get("seed"))
    check("smc.lossScale", smc.get("lossScale"), float(config.smc.loss_scale))
    check("smc.costScale", smc.get("costScale"), float(config.smc.cost_scale))
    check("result.deduction_strength", result.get("deduction_strength"), expected_strength)
    check("result.mode", result.get("mode"), "importance-smc")
    check(
        "result.proposal_source",
        result.get("proposal_source"),
        expected_proposal_source,
    )
    check(
        "result.llm_energy_normalization",
        result.get("llm_energy_normalization"),
        expected_normalization,
    )
    check("result.max_scored_candidates", result.get("max_scored_candidates"), expected_score_cap)
    check(
        "result.stage_count",
        len(_array(result.get("stages"), name="result.stages")),
        expected_iterations,
    )
    check(
        "result.reference_materialized",
        result.get("reference") is not None,
        expected_materialize,
    )

    model_id = cell_plan.get("model_id")
    expected_model_alias = (
        None if model_id is None else cell_plan.get("model_alias")
    )
    expected_ledger_model = "none" if expected_model_alias is None else expected_model_alias
    check("runtime.model", runtime.get("model"), expected_model_alias)
    check(
        "runtime.model_repository",
        runtime.get("model_repository"),
        cell_plan.get("model_hf_repository"),
    )
    check("runtime.model_revision", runtime.get("model_revision"), cell_plan.get("model_revision"))
    check(
        "runtime.tokenizer_revision",
        runtime.get("tokenizer_revision"),
        cell_plan.get("tokenizer_revision"),
    )

    for index, ledger in enumerate(ledgers):
        prefix = f"ledger[{index}]"
        check(f"{prefix}.stage", ledger.get("stage"), 1)
        check(f"{prefix}.beta", ledger.get("beta"), expected_beta)
        check(f"{prefix}.temperature", ledger.get("temperature"), expected_temperature)
        check(
            f"{prefix}.proposal_epsilon",
            ledger.get("proposal_epsilon"),
            expected_epsilon,
        )
        check(
            f"{prefix}.energy_normalization",
            ledger.get("energy_normalization"),
            expected_normalization,
        )
        check(f"{prefix}.model", ledger.get("model"), expected_ledger_model)
        check(
            f"{prefix}.model_revision",
            ledger.get("model_revision"),
            cell_plan.get("model_revision"),
        )
        check(
            f"{prefix}.tokenizer_revision",
            ledger.get("tokenizer_revision"),
            cell_plan.get("tokenizer_revision"),
        )

    raw_command = core_manifest.get("command")
    if cell_document.get("command") != raw_command:
        mismatches.append("cell.command differs from manifest.command")
    if not isinstance(raw_command, list) or not all(
        isinstance(value, str) for value in raw_command
    ):
        mismatches.append("manifest.command must be an array of strings")
    else:
        command = cast(list[str], raw_command)

        def command_value(flag: str) -> str | None:
            positions = [index for index, value in enumerate(command) if value == flag]
            if len(positions) != 1 or positions[0] + 1 >= len(command):
                mismatches.append(f"command flag {flag} must occur exactly once with a value")
                return None
            return command[positions[0] + 1]

        if len(command) < 3 or command[1] != "synthesize":
            mismatches.append("command must invoke the synthesize subcommand")
        else:
            task = _mapping(
                _array(protocol.get("tasks"), name="protocol.tasks")[0],
                name="protocol task",
            )
            check_path_identity("command spec", command[2], task.get("spec"))
        expected_command_model = (
            cell_plan.get("model_alias")
            if provider_backed
            else "size-invariant-catalog-control"
        )
        command_expectations: tuple[tuple[str, str], ...] = (
            ("--mode", "importance-smc"),
            ("--proposal", str(arm.get("proposal"))),
            ("--model", str(expected_command_model)),
            ("--skeleton", "auto"),
            ("--particles", str(expected_particles)),
            ("--iterations", str(expected_iterations)),
            ("--alpha", str(expected_alpha)),
            ("--ess-threshold", str(expected_ess)),
            ("--seed", str(cell_plan.get("seed"))),
            ("--max-scored-candidates", str(expected_score_cap)),
            ("--support-limit", str(caps.get("support_limit"))),
            ("--hole-state-limit", str(caps.get("hole_state_limit"))),
            ("--hole-max-cost", str(caps.get("hole_max_cost"))),
            ("--timeout-seconds", str(expected_timeout)),
            ("--deduction-mix", str(fallback_mix)),
            ("--beta-max", str(expected_beta)),
            ("--candidate-batch-size", str(shared.get("candidate_batch_size"))),
            ("--deduction-strength", str(expected_strength)),
            ("--llm-energy-normalization", str(expected_normalization)),
            ("--proposal-epsilon", str(expected_epsilon)),
            ("--temperature", str(expected_temperature)),
        )
        for flag, expected in command_expectations:
            observed = command_value(flag)
            if observed is not None and observed != expected:
                mismatches.append(f"command {flag}={observed!r}, expected={expected!r}")
        if "family_deduction_mix" in arm:
            observed = command_value("--family-deduction-mix")
            if observed is not None and observed != str(family_mix):
                mismatches.append(
                    f"command --family-deduction-mix={observed!r}, expected={family_mix!r}"
                )
        if "hole_deduction_mix" in arm:
            observed = command_value("--hole-deduction-mix")
            if observed is not None and observed != str(hole_mix):
                mismatches.append(
                    f"command --hole-deduction-mix={observed!r}, expected={hole_mix!r}"
                )
        provider_flags = (
            "--model-repository",
            "--model-revision",
            "--tokenizer-revision",
            "--base-url",
            "--score-cache-dir",
            "--score-cache-mode",
            "--vllm-server-config",
            "--api-key-env",
        )
        if provider_backed:
            for flag, expected_value in (
                ("--model-repository", cell_plan.get("model_hf_repository")),
                ("--model-revision", cell_plan.get("model_revision")),
                ("--tokenizer-revision", cell_plan.get("tokenizer_revision")),
            ):
                observed = command_value(flag)
                if observed is not None and observed != expected_value:
                    mismatches.append(
                        f"command {flag}={observed!r}, expected={expected_value!r}"
                    )
            base_url = command_value("--base-url")
            if base_url is not None and not base_url.strip():
                mismatches.append("provider command --base-url must be nonempty")
            if expected_cache_mode == "off":
                for flag in (
                    "--score-cache-dir",
                    "--score-cache-mode",
                    "--vllm-server-config",
                ):
                    if flag in command:
                        mismatches.append(f"provider command unexpectedly contains {flag}")
            else:
                observed_cache_mode = command_value("--score-cache-mode")
                if (
                    observed_cache_mode is not None
                    and observed_cache_mode != expected_cache_mode
                ):
                    mismatches.append(
                        "command --score-cache-mode="
                        f"{observed_cache_mode!r}, expected={expected_cache_mode!r}"
                    )
                observed_server_config = command_value("--vllm-server-config")
                if (
                    observed_server_config is not None
                    and observed_server_config != expected_server_config
                ):
                    mismatches.append(
                        "command --vllm-server-config differs from frozen provider config"
                    )
                observed_cache_dir = command_value("--score-cache-dir")
                if observed_cache_dir is not None:
                    check_path_identity(
                        "command --score-cache-dir",
                        observed_cache_dir,
                        expected_cache_dir,
                    )
            expected_api_key_env = provider_document.get("api_key_env")
            if expected_api_key_env is None:
                if "--api-key-env" in command:
                    mismatches.append("provider command unexpectedly contains --api-key-env")
            else:
                observed_api_key_env = command_value("--api-key-env")
                if (
                    observed_api_key_env is not None
                    and observed_api_key_env != expected_api_key_env
                ):
                    mismatches.append(
                        f"command --api-key-env={observed_api_key_env!r}, "
                        f"expected={expected_api_key_env!r}"
                    )
        else:
            for flag in provider_flags:
                if flag in command:
                    mismatches.append(f"catalog command unexpectedly contains {flag}")
        materialize_count = command.count("--materialize-reference")
        if materialize_count != int(expected_materialize):
            mismatches.append(
                "command --materialize-reference count="
                f"{materialize_count}, expected={int(expected_materialize)}"
            )

    if expected_iterations != 1 or expected_alpha != 0.0:
        mismatches.append(
            "terminal ledger/path replay requires protocol iterations=1 and alpha=0"
        )
    if mismatches:
        _violation(
            violations,
            "PROTOCOL_CONFIGURATION_MISMATCH",
            "; ".join(mismatches),
            cell_id=cell_id,
        )


def _audit_ledger_particle_paths(
    particles: Sequence[Mapping[str, Any]],
    ledgers: Sequence[Mapping[str, Any]],
    support: FactorizedImportanceSupport,
    *,
    allowed_integer_constants: Sequence[int],
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Reconstruct each alpha-zero proposal path and its exact ledger product."""

    choices_by_slot: dict[int, dict[str, str]] = {}
    log_q_by_slot: dict[int, float] = {}
    ancestor_by_slot: dict[int, object] = {}
    for ledger in ledgers:
        wave = ledger.get("wave")
        if not isinstance(wave, str):
            continue
        candidates = _array(ledger.get("candidates"), name="ledger candidates")
        for raw_selection in _array(ledger.get("selections"), name="ledger selections"):
            selection = _mapping(raw_selection, name="ledger selection")
            slot = _integer(selection.get("slot"), name="selection.slot")
            if selection.get("forced") is not False or selection.get("cloned") is not False:
                _violation(
                    violations,
                    "UNSUPPORTED_FORCED_OR_CLONED_PATH",
                    f"slot={slot}, wave={wave}",
                    cell_id=cell_id,
                )
                continue
            selection_ancestor = selection.get("ancestor_state_index")
            if slot in ancestor_by_slot and ancestor_by_slot[slot] != selection_ancestor:
                _violation(
                    violations,
                    "SLOT_ANCESTOR_INCONSISTENT",
                    (
                        f"slot={slot}, first={ancestor_by_slot[slot]!r}, "
                        f"observed={selection_ancestor!r}"
                    ),
                    cell_id=cell_id,
                )
            ancestor_by_slot[slot] = selection_ancestor
            selected_index = _integer(
                selection.get("selected_index"), name="selection.selected_index"
            )
            if not 0 <= selected_index < len(candidates):
                continue
            candidate = _mapping(candidates[selected_index], name="selected candidate")
            canonical = candidate.get("canonical_candidate")
            if not isinstance(canonical, str):
                continue
            slot_choices = choices_by_slot.setdefault(slot, {})
            if wave in slot_choices:
                _violation(
                    violations,
                    "DUPLICATE_SLOT_WAVE_SELECTION",
                    f"slot={slot}, wave={wave}",
                    cell_id=cell_id,
                )
            slot_choices[wave] = canonical
            probability = _number(
                selection.get("selected_probability"),
                name="selection.selected_probability",
            )
            if probability <= 0.0:
                _violation(
                    violations,
                    "SELECTED_PROBABILITY_NOT_POSITIVE",
                    f"slot={slot}, wave={wave}, probability={probability}",
                    cell_id=cell_id,
                )
                continue
            log_q_by_slot[slot] = log_q_by_slot.get(slot, 0.0) + math.log(probability)

    family_by_name = {
        family.hypothesis.kind.value: family for family in support.families
    }
    seen_particle_slots: set[int] = set()
    for particle in particles:
        slot = _integer(particle.get("particle_index"), name="particle.particle_index")
        if slot in seen_particle_slots:
            _violation(
                violations,
                "DUPLICATE_PARTICLE_SLOT",
                f"slot={slot}",
                cell_id=cell_id,
            )
        seen_particle_slots.add(slot)
        if particle.get("cloned") is not False:
            _violation(
                violations,
                "PARTICLE_CLONE_FLAG_INVALID",
                f"slot={slot}, cloned={particle.get('cloned')!r}",
                cell_id=cell_id,
            )
        if "ancestor_state_index" in particle and (
            ancestor_by_slot.get(slot) != particle.get("ancestor_state_index")
        ):
            _violation(
                violations,
                "PARTICLE_LEDGER_ANCESTOR_MISMATCH",
                (
                    f"slot={slot}, ledger={ancestor_by_slot.get(slot)!r}, "
                    f"particle={particle.get('ancestor_state_index')!r}"
                ),
                cell_id=cell_id,
            )
        choices = choices_by_slot.get(slot)
        if choices is None or slot not in log_q_by_slot:
            _violation(
                violations,
                "PARTICLE_LEDGER_PATH_MISSING",
                f"slot={slot}",
                cell_id=cell_id,
            )
            continue
        stored_log_q = _number(
            particle.get("log_q_mixture"), name="particle.log_q_mixture"
        )
        replayed_log_q = log_q_by_slot[slot]
        if not math.isclose(
            replayed_log_q, stored_log_q, rel_tol=1e-12, abs_tol=1e-12
        ):
            _violation(
                violations,
                "PARTICLE_LOG_Q_LEDGER_MISMATCH",
                f"slot={slot}, replayed={replayed_log_q}, stored={stored_log_q}",
                cell_id=cell_id,
            )

        family_key = choices.get("family")
        try:
            family_name = json.loads(family_key) if isinstance(family_key, str) else None
        except json.JSONDecodeError:
            family_name = None
        if not isinstance(family_name, str) or family_name not in family_by_name:
            _violation(
                violations,
                "LEDGER_FAMILY_SELECTION_INVALID",
                f"slot={slot}, family={family_key!r}",
                cell_id=cell_id,
            )
            continue
        family = family_by_name[family_name]
        expected_waves = {"family"} | {
            f"hole-{index}" for index in range(len(family.catalogs))
        }
        if set(choices) != expected_waves:
            _violation(
                violations,
                "LEDGER_PATH_WAVE_SET_MISMATCH",
                (
                    f"slot={slot}, expected={sorted(expected_waves)}, "
                    f"observed={sorted(choices)}"
                ),
                cell_id=cell_id,
            )
            continue
        filling_indices: list[int] = []
        for hole_index, catalog in enumerate(family.catalogs):
            selected_key = choices[f"hole-{hole_index}"]
            indices = [
                index
                for index, filling in enumerate(catalog.fillings)
                if filling.key == selected_key
            ]
            if len(indices) != 1:
                _violation(
                    violations,
                    "LEDGER_HOLE_SELECTION_INVALID",
                    f"slot={slot}, hole={hole_index}, matches={len(indices)}",
                    cell_id=cell_id,
                )
                break
            filling_indices.append(indices[0])
        if len(filling_indices) != len(family.catalogs):
            continue
        trace = ConstructionTrace(
            hypothesis_index=family.hypothesis_index,
            filling_indices=tuple(filling_indices),
        )
        fillings = trace_fillings(family, trace)
        reconstructed = assemble_program(
            family.hypothesis,
            {filling.hole_name: filling.expression for filling in fillings},
            allowed_integer_constants=allowed_integer_constants,
        )
        raw_program = particle.get("program")
        if not isinstance(raw_program, dict):
            raise ValueError("particle.program must be an object")
        if canonical_key(reconstructed) != canonical_key(cast(ProgramAst, raw_program)):
            _violation(
                violations,
                "PARTICLE_PROGRAM_LEDGER_PATH_MISMATCH",
                f"slot={slot}",
                cell_id=cell_id,
            )
        if particle.get("family") != family_name:
            _violation(
                violations,
                "PARTICLE_FAMILY_LEDGER_MISMATCH",
                f"slot={slot}, ledger={family_name}, particle={particle.get('family')}",
                cell_id=cell_id,
            )

    expected_slots = set(range(len(particles)))
    if seen_particle_slots != expected_slots:
        _violation(
            violations,
            "PARTICLE_SLOT_DOMAIN_MISMATCH",
            f"expected={sorted(expected_slots)}, observed={sorted(seen_particle_slots)}",
            cell_id=cell_id,
        )
    if set(choices_by_slot) != expected_slots:
        _violation(
            violations,
            "LEDGER_PARTICLE_SLOT_SET_MISMATCH",
            (
                f"ledger_slots={sorted(choices_by_slot)}, "
                f"expected_slots={sorted(expected_slots)}"
            ),
            cell_id=cell_id,
        )


def _audit_catalog_and_deduction_guides(
    ledgers: Sequence[Mapping[str, Any]],
    support: FactorizedImportanceSupport,
    config: ExperimentConfig,
    options: ImportanceSMCOptions,
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Rebuild every finite catalog and deduction guide from its selected prefix."""

    family_guide = family_guide_probabilities(
        support,
        cost_scale=float(config.smc.cost_scale),
        violation_scale=options.deduction_strength,
    )
    family_by_name = {
        family.hypothesis.kind.value: family for family in support.families
    }
    sorted_families = tuple(
        sorted(support.families, key=lambda family: family.hypothesis.kind.value)
    )
    family_guide_by_index = {
        family.hypothesis_index: float(family_guide[index])
        for index, family in enumerate(support.families)
    }
    selected_family: dict[int, FactorizedFamily] = {}
    selected_indices: dict[int, list[int]] = {}
    prefix_costs: dict[int, int] = {}

    for ledger in ledgers:
        wave = ledger.get("wave")
        candidates = [
            _mapping(candidate, name="ledger candidate")
            for candidate in _array(ledger.get("candidates"), name="ledger candidates")
        ]
        selections = [
            _mapping(selection, name="ledger selection")
            for selection in _array(ledger.get("selections"), name="ledger selections")
        ]
        expected_keys: list[str] | None = None
        expected_guide: tuple[float, ...] | None = None
        choice_indices_by_slot: dict[int, tuple[int, ...]] = {}

        if wave == "family":
            expected_keys = [
                json.dumps(family.hypothesis.kind.value) for family in sorted_families
            ]
            expected_guide = tuple(
                family_guide_by_index[family.hypothesis_index]
                for family in sorted_families
            )
        elif isinstance(wave, str) and wave.startswith("hole-"):
            try:
                hole_index = int(wave.removeprefix("hole-"))
            except ValueError:
                hole_index = -1
            for selection in selections:
                slot = _integer(selection.get("slot"), name="selection.slot")
                family = selected_family.get(slot)
                indices_so_far = selected_indices.get(slot, [])
                if family is None or hole_index != len(indices_so_far):
                    _violation(
                        violations,
                        "LEDGER_PREFIX_ORDER_INVALID",
                        f"slot={slot}, wave={wave}",
                        cell_id=cell_id,
                    )
                    continue
                choices = valid_choice_indices(
                    family,
                    hole_index=hole_index,
                    prefix_cost=prefix_costs.get(slot, 0),
                )
                keys = [
                    family.catalogs[hole_index].fillings[index].key
                    for index in choices
                ]
                guide = choice_guide_probabilities(
                    family,
                    hole_index=hole_index,
                    prefix_cost=prefix_costs.get(slot, 0),
                    choice_indices=choices,
                    cost_scale=float(config.smc.cost_scale),
                    violation_scale=options.deduction_strength,
                )
                if expected_keys is None:
                    expected_keys = keys
                    expected_guide = tuple(float(value) for value in guide)
                elif expected_keys != keys or expected_guide != tuple(
                    float(value) for value in guide
                ):
                    _violation(
                        violations,
                        "DEDUPLICATED_LEDGER_PREFIX_MISMATCH",
                        f"wave={wave}",
                        cell_id=cell_id,
                    )
                choice_indices_by_slot[slot] = choices

        if expected_keys is None or expected_guide is None:
            _violation(
                violations,
                "LEDGER_CATALOG_NOT_RECONSTRUCTED",
                f"wave={wave!r}",
                cell_id=cell_id,
            )
            continue
        stored_keys = [str(candidate.get("canonical_candidate")) for candidate in candidates]
        if stored_keys != expected_keys:
            _violation(
                violations,
                "LEDGER_CATALOG_REPLAY_MISMATCH",
                f"wave={wave}, expected={len(expected_keys)}, stored={len(stored_keys)}",
                cell_id=cell_id,
            )
        stored_guide = tuple(
            _number(
                candidate.get("deduction_probability"),
                name="deduction_probability",
            )
            for candidate in candidates
        )
        if len(stored_guide) != len(expected_guide) or any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(stored_guide, expected_guide, strict=False)
        ):
            _violation(
                violations,
                "DEDUCTION_GUIDE_REPLAY_MISMATCH",
                f"wave={wave}",
                cell_id=cell_id,
            )

        for selection in selections:
            slot = _integer(selection.get("slot"), name="selection.slot")
            selected_index = _integer(
                selection.get("selected_index"), name="selection.selected_index"
            )
            if not 0 <= selected_index < len(candidates):
                continue
            if wave == "family":
                try:
                    family_name = json.loads(expected_keys[selected_index])
                except json.JSONDecodeError:
                    continue
                family = family_by_name.get(family_name)
                if family is not None:
                    selected_family[slot] = family
                    selected_indices[slot] = []
                    prefix_costs[slot] = 0
            elif isinstance(wave, str) and wave.startswith("hole-"):
                family = selected_family.get(slot)
                selected_choices = choice_indices_by_slot.get(slot)
                if (
                    family is None
                    or selected_choices is None
                    or selected_index >= len(selected_choices)
                ):
                    continue
                global_index = selected_choices[selected_index]
                hole_index = len(selected_indices[slot])
                selected_indices[slot].append(global_index)
                prefix_costs[slot] += family.catalogs[hole_index].costs[global_index]


def _trace_from_json(value: object, *, name: str) -> ConstructionTrace:
    record = _mapping(value, name=name)
    return ConstructionTrace(
        hypothesis_index=_integer(
            record.get("hypothesis_index"), name=f"{name}.hypothesis_index"
        ),
        filling_indices=tuple(
            _integer(index, name=f"{name}.filling_index")
            for index in _array(
                record.get("filling_indices"), name=f"{name}.filling_indices"
            )
        ),
    )


def _audit_score_prompt_provenance(
    particles: Sequence[Mapping[str, Any]],
    ledgers: Sequence[Mapping[str, Any]],
    support: FactorizedImportanceSupport,
    runtime_support: ImportanceSupport,
    config: ExperimentConfig,
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Bind every stored score prompt to its particle ancestor and hole prefix."""

    particle_by_slot = {
        _integer(particle.get("particle_index"), name="particle.particle_index"): particle
        for particle in particles
    }
    lazy_shape = all(
        "trace" in particle and "ancestor_trace" in particle for particle in particles
    )
    materialized_shape = all(
        "state_index" in particle and "ancestor_state_index" in particle
        for particle in particles
    )
    if lazy_shape == materialized_shape:
        _violation(
            violations,
            "PROMPT_PREFIX_PROVENANCE_UNBOUND",
            "particle result shape is mixed or incomplete",
            cell_id=cell_id,
        )
        return

    runtime_by_key = {state.key: state for state in runtime_support.states}
    lazy_ancestors: dict[int, LazyImportanceState] = {}
    materialized_ancestors: dict[int, ImportanceState] = {}
    for slot, particle_record in particle_by_slot.items():
        if lazy_shape:
            trace = _trace_from_json(
                particle_record.get("ancestor_trace"), name="particle.ancestor_trace"
            )
            try:
                ancestor_family = support.family(trace.hypothesis_index)
                fillings = trace_fillings(ancestor_family, trace)
                program = assemble_program(
                    ancestor_family.hypothesis,
                    {
                        filling.hole_name: filling.expression
                        for filling in fillings
                    },
                    allowed_integer_constants=config.spec.integer_constants,
                )
                key = canonical_key(program)
                runtime_state = runtime_by_key[key]
            except (IndexError, KeyError, ValueError):
                _violation(
                    violations,
                    "PROMPT_PREFIX_PROVENANCE_UNBOUND",
                    f"slot={slot}, invalid lazy ancestor trace",
                    cell_id=cell_id,
                )
                continue
            lazy_ancestors[slot] = LazyImportanceState(
                trace=trace,
                family=ancestor_family.hypothesis.kind.value,
                fillings=fillings,
                program=program,
                key=key,
                score=runtime_state.score,
                log_prior=trace_log_prior(
                    support,
                    trace,
                    cost_scale=float(config.smc.cost_scale),
                ),
            )
        else:
            ancestor_index = _integer(
                particle_record.get("ancestor_state_index"),
                name="particle.ancestor_state_index",
            )
            if not 0 <= ancestor_index < len(runtime_support.states):
                _violation(
                    violations,
                    "PROMPT_PREFIX_PROVENANCE_UNBOUND",
                    f"slot={slot}, ancestor_state_index={ancestor_index}",
                    cell_id=cell_id,
                )
                continue
            materialized_ancestors[slot] = runtime_support.states[ancestor_index]

    factorized_families = tuple(
        sorted(support.families, key=lambda family: family.hypothesis.kind.value)
    )
    materialized_families = tuple(
        sorted(runtime_support.families, key=family_candidate)
    )
    family_by_name = {
        family.hypothesis.kind.value: family for family in support.families
    }
    runtime_family_by_hypothesis: dict[int, FamilySupport] = {
        family.hypothesis_index: family for family in runtime_support.families
    }
    selected_family: dict[int, FactorizedFamily] = {}
    previous_fillings: dict[int, list[HoleFilling]] = {}

    def unbound(*, ledger_index: int, wave: object, slot: int, detail: str) -> None:
        _violation(
            violations,
            "PROMPT_PREFIX_PROVENANCE_UNBOUND",
            f"ledger={ledger_index}, wave={wave!r}, slot={slot}: {detail}",
            cell_id=cell_id,
        )

    for ledger_index, ledger in enumerate(ledgers):
        wave = ledger.get("wave")
        stage = _integer(ledger.get("stage"), name="ledger.stage")
        beta = _number(ledger.get("beta"), name="ledger.beta")
        candidates = [
            _mapping(candidate, name="ledger candidate")
            for candidate in _array(ledger.get("candidates"), name="ledger candidates")
        ]
        stored_prefix = ledger.get("prompt_prefix")
        for raw_selection in _array(
            ledger.get("selections"), name="ledger selections"
        ):
            selection = _mapping(raw_selection, name="ledger selection")
            slot = _integer(selection.get("slot"), name="selection.slot")
            selected_index = _integer(
                selection.get("selected_index"), name="selection.selected_index"
            )
            slot_particle = particle_by_slot.get(slot)
            ancestor: LazyImportanceState | ImportanceState | None
            ancestor = (
                lazy_ancestors.get(slot)
                if lazy_shape
                else materialized_ancestors.get(slot)
            )
            if slot_particle is None or ancestor is None:
                unbound(
                    ledger_index=ledger_index,
                    wave=wave,
                    slot=slot,
                    detail="particle ancestor is unavailable",
                )
                continue
            if not 0 <= selected_index < len(candidates):
                unbound(
                    ledger_index=ledger_index,
                    wave=wave,
                    slot=slot,
                    detail="selected candidate is out of range",
                )
                continue

            expected_prefix: str | None = None
            path_family: FactorizedFamily | None = None
            hole_index: int | None = None
            if wave == "family":
                if lazy_shape:
                    assert isinstance(ancestor, LazyImportanceState)
                    expected_prefix = lazy_family_prompt_prefix(
                        config=config,
                        families=factorized_families,
                        ancestor=ancestor,
                        stage=stage,
                        beta=beta,
                    )
                else:
                    assert isinstance(ancestor, ImportanceState)
                    expected_prefix = family_prompt_prefix(
                        config=config,
                        families=materialized_families,
                        ancestor=ancestor,
                        stage=stage,
                        beta=beta,
                    )
            elif isinstance(wave, str) and wave.startswith("hole-"):
                try:
                    hole_index = int(wave.removeprefix("hole-"))
                except ValueError:
                    hole_index = None
                path_family = selected_family.get(slot)
                prior = previous_fillings.get(slot)
                if (
                    hole_index is None
                    or path_family is None
                    or prior is None
                    or hole_index != len(prior)
                    or not 0 <= hole_index < len(path_family.catalogs)
                ):
                    unbound(
                        ledger_index=ledger_index,
                        wave=wave,
                        slot=slot,
                        detail="selected family or previous fillings are unavailable",
                    )
                    continue
                if lazy_shape:
                    assert isinstance(ancestor, LazyImportanceState)
                    expected_prefix = lazy_hole_prompt_prefix(
                        config=config,
                        family=path_family,
                        hole_index=hole_index,
                        ancestor=ancestor,
                        previous_fillings=tuple(prior),
                        stage=stage,
                        beta=beta,
                        candidate_count=len(candidates),
                    )
                else:
                    assert isinstance(ancestor, ImportanceState)
                    runtime_family = runtime_family_by_hypothesis.get(
                        path_family.hypothesis_index
                    )
                    if runtime_family is None:
                        unbound(
                            ledger_index=ledger_index,
                            wave=wave,
                            slot=slot,
                            detail="runtime family is unavailable",
                        )
                        continue
                    expected_prefix = hole_prompt_prefix(
                        config=config,
                        report=runtime_family.deduction,
                        hole=runtime_family.hypothesis.holes[hole_index],
                        ancestor=ancestor,
                        previous_fillings=tuple(prior),
                        stage=stage,
                        beta=beta,
                        candidate_count=len(candidates),
                    )
            else:
                unbound(
                    ledger_index=ledger_index,
                    wave=wave,
                    slot=slot,
                    detail="unknown wave",
                )
                continue

            if stored_prefix != expected_prefix:
                expected_digest = hashlib.sha256(expected_prefix.encode()).hexdigest()
                stored_digest = (
                    hashlib.sha256(stored_prefix.encode()).hexdigest()
                    if isinstance(stored_prefix, str)
                    else "not-a-string"
                )
                _violation(
                    violations,
                    "PROMPT_PREFIX_PROVENANCE_MISMATCH",
                    (
                        f"ledger={ledger_index}, wave={wave!r}, slot={slot}, "
                        f"expected_sha256={expected_digest}, "
                        f"stored_sha256={stored_digest}"
                    ),
                    cell_id=cell_id,
                )

            selected_key = candidates[selected_index].get("canonical_candidate")
            if not isinstance(selected_key, str):
                continue
            if wave == "family":
                try:
                    family_name = json.loads(selected_key)
                except json.JSONDecodeError:
                    continue
                selected = family_by_name.get(family_name)
                if selected is not None:
                    selected_family[slot] = selected
                    previous_fillings[slot] = []
            elif path_family is not None and hole_index is not None:
                matches = [
                    filling
                    for filling in path_family.catalogs[hole_index].fillings
                    if filling.key == selected_key
                ]
                if len(matches) == 1:
                    previous_fillings[slot].append(matches[0])


def _audit_final_particle_math(
    particles: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    runtime_support: ImportanceSupport,
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Reconcile persisted terminal particles with the enumerated target."""

    by_trace = {
        cast(ConstructionTrace, state["trace_record"]): state for state in states
    }
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for state in states:
        by_key.setdefault(str(state["key"]), []).append(state)
    weights: list[float] = []
    log_incremental_weights: list[float] = []
    for particle in particles:
        raw_program = particle.get("program")
        if not isinstance(raw_program, dict):
            raise ValueError("particle.program must be an object")
        program_key = canonical_key(cast(ProgramAst, raw_program))
        if "state_index" in particle:
            state_index = _integer(
                particle.get("state_index"), name="particle.state_index"
            )
            if not 0 <= state_index < len(runtime_support.states):
                _violation(
                    violations,
                    "PARTICLE_STATE_INDEX_OUT_OF_RANGE",
                    f"state_index={state_index}, support={len(runtime_support.states)}",
                    cell_id=cell_id,
                )
                continue
            ancestor_state_index = _integer(
                particle.get("ancestor_state_index"),
                name="particle.ancestor_state_index",
            )
            if not 0 <= ancestor_state_index < len(runtime_support.states):
                _violation(
                    violations,
                    "PARTICLE_ANCESTOR_INDEX_OUT_OF_RANGE",
                    (
                        f"ancestor_state_index={ancestor_state_index}, "
                        f"support={len(runtime_support.states)}"
                    ),
                    cell_id=cell_id,
                )
            runtime_state = runtime_support.states[state_index]
            if runtime_state.key != program_key:
                _violation(
                    violations,
                    "PARTICLE_STATE_INDEX_PROGRAM_MISMATCH",
                    f"state_index={state_index}",
                    cell_id=cell_id,
                )
            matching_states = by_key.get(program_key, [])
            if not matching_states:
                _violation(
                    violations,
                    "PARTICLE_PROGRAM_NOT_IN_SUPPORT",
                    f"state_index={state_index}",
                    cell_id=cell_id,
                )
                continue
            if len(matching_states) != 1:
                _violation(
                    violations,
                    "PARTICLE_PROGRAM_SUPPORT_AMBIGUOUS",
                    f"state_index={state_index}, matching_traces={len(matching_states)}",
                    cell_id=cell_id,
                )
                continue
            state = matching_states[0]
            if (
                runtime_state.score.total_loss != state["loss"]
                or runtime_state.score.cost != state["cost"]
                or runtime_state.score.exact_program is not state["exact"]
            ):
                _violation(
                    violations,
                    "RUNTIME_FACTORIZED_SUPPORT_MISMATCH",
                    f"state_index={state_index}",
                    cell_id=cell_id,
                )
        else:
            raw_trace = _mapping(particle.get("trace"), name="particle.trace")
            trace = ConstructionTrace(
                hypothesis_index=_integer(
                    raw_trace.get("hypothesis_index"), name="trace.hypothesis_index"
                ),
                filling_indices=tuple(
                    _integer(value, name="trace filling index")
                    for value in _array(
                        raw_trace.get("filling_indices"), name="trace.filling_indices"
                    )
                ),
            )
            traced_state = by_trace.get(trace)
            if traced_state is None:
                _violation(
                    violations,
                    "PARTICLE_TRACE_NOT_IN_SUPPORT",
                    f"trace={trace}",
                    cell_id=cell_id,
                )
                continue
            state = traced_state

        if program_key != state["key"]:
            _violation(
                violations,
                "PARTICLE_PATH_PROGRAM_MISMATCH",
                "persisted path identity resolves to a different program",
                cell_id=cell_id,
            )
        stored_loss = _number(particle.get("total_loss"), name="particle.total_loss")
        stored_cost = _integer(particle.get("cost"), name="particle.cost")
        stored_exact = particle.get("exact_program")
        if not isinstance(stored_exact, bool):
            raise ValueError("particle.exact_program must be a boolean")
        if (
            not math.isclose(stored_loss, float(state["loss"]), rel_tol=0.0, abs_tol=1e-12)
            or stored_cost != state["cost"]
            or stored_exact is not state["exact"]
        ):
            _violation(
                violations,
                "PARTICLE_STATE_SUMMARY_MISMATCH",
                (
                    f"loss={stored_loss}/{state['loss']}, cost={stored_cost}/{state['cost']}, "
                    f"exact={stored_exact}/{state['exact']}"
                ),
                cell_id=cell_id,
            )

        log_q = _number(particle.get("log_q_mixture"), name="particle.log_q_mixture")
        log_incremental = _number(
            particle.get("log_incremental_weight"),
            name="particle.log_incremental_weight",
        )
        expected_log_target = float(state["log_target"])
        if not math.isclose(
            log_q + log_incremental,
            expected_log_target,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            _violation(
                violations,
                "PARTICLE_IMPORTANCE_IDENTITY_MISMATCH",
                (
                    f"log_q+log_incremental={log_q + log_incremental}, "
                    f"log_target={expected_log_target}"
                ),
                cell_id=cell_id,
            )
        weights.append(_number(particle.get("weight"), name="particle.weight"))
        log_incremental_weights.append(log_incremental)

    if len(weights) != len(particles):
        return
    if any(weight < 0.0 for weight in weights) or not math.isclose(
        math.fsum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        _violation(
            violations,
            "FINAL_PARTICLE_WEIGHTS_INVALID",
            f"sum={math.fsum(weights)}",
            cell_id=cell_id,
        )
    maximum = max(log_incremental_weights)
    unnormalized = [math.exp(value - maximum) for value in log_incremental_weights]
    normalizer = math.fsum(unnormalized)
    replayed_weights = [value / normalizer for value in unnormalized]
    if any(
        not math.isclose(actual, stored, rel_tol=1e-12, abs_tol=1e-12)
        for actual, stored in zip(replayed_weights, weights, strict=True)
    ):
        _violation(
            violations,
            "FINAL_PARTICLE_WEIGHT_REPLAY_MISMATCH",
            "normalized incremental weights do not reproduce final weights",
            cell_id=cell_id,
        )


def _audit_finite_reference(
    result: Mapping[str, Any],
    support_math: Mapping[str, object],
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Check exact-reference fields when the finite result schema persists them."""

    raw_reference = result.get("reference")
    if raw_reference is None:
        return
    reference = _mapping(raw_reference, name="result.reference")
    equations = (
        (
            "enumeration_exact_mass",
            _number(support_math.get("exact_target_mass"), name="exact_target_mass"),
        ),
        (
            "log_path_z_enumeration",
            _number(
                support_math.get("log_target_normalizer"),
                name="log_target_normalizer",
            ),
        ),
    )
    for field, expected in equations:
        stored = _number(reference.get(field), name=f"reference.{field}")
        if not math.isclose(expected, stored, rel_tol=1e-12, abs_tol=1e-12):
            _violation(
                violations,
                "FINITE_REFERENCE_REPLAY_MISMATCH",
                f"{field}: expected={expected}, stored={stored}",
                cell_id=cell_id,
            )


def _audit_sampled_best(
    result: Mapping[str, Any],
    particles: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    runtime_support: ImportanceSupport,
    support_math: Mapping[str, object],
    *,
    cell_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Recompute the materialized result's terminal champion from its particles."""

    if "search" in result:
        return
    state_by_key = {str(state["key"]): state for state in states}
    weights_by_index: dict[int, float] = {}
    for particle in particles:
        state_index = _integer(particle.get("state_index"), name="particle.state_index")
        weights_by_index[state_index] = weights_by_index.get(state_index, 0.0) + _number(
            particle.get("weight"), name="particle.weight"
        )
    if not weights_by_index:
        raise ValueError("materialized result has no final particles")

    def ordering(state_index: int) -> tuple[float, str]:
        key = runtime_support.states[state_index].key
        state = state_by_key.get(key)
        if state is None:
            raise ValueError("runtime state is absent from factorized support")
        return (-_number(state.get("log_target"), name="state.log_target"), key)

    best_index = min(weights_by_index, key=ordering)
    best_state = runtime_support.states[best_index]
    factorized_state = state_by_key[best_state.key]
    summary = _mapping(result.get("sampled_best"), name="result.sampled_best")
    raw_program = summary.get("program")
    if not isinstance(raw_program, dict):
        raise ValueError("sampled_best.program must be an object")
    expected_enumeration_probability = math.exp(
        _number(factorized_state.get("log_target"), name="state.log_target")
        - _number(
            support_math.get("log_target_normalizer"),
            name="log_target_normalizer",
        )
    )
    equations = (
        summary.get("state_index") == best_index,
        summary.get("family") == best_state.family,
        canonical_key(cast(ProgramAst, raw_program)) == best_state.key,
        math.isclose(
            _number(summary.get("total_loss"), name="sampled_best.total_loss"),
            best_state.score.total_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        summary.get("cost") == best_state.score.cost,
        summary.get("exact_program") is best_state.score.exact_program,
        math.isclose(
            _number(summary.get("empirical_mass"), name="sampled_best.empirical_mass"),
            weights_by_index[best_index],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        math.isclose(
            _number(
                summary.get("enumeration_probability"),
                name="sampled_best.enumeration_probability",
            ),
            expected_enumeration_probability,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )
    if not all(equations):
        _violation(
            violations,
            "SAMPLED_BEST_REPLAY_MISMATCH",
            f"expected_state_index={best_index}",
            cell_id=cell_id,
        )


def audit_matrix(matrix_dir: Path) -> dict[str, object]:
    """Run the end-to-end read-only audit for one sealed deduction-stress matrix."""

    matrix_dir = matrix_dir.expanduser().resolve()
    violations: list[dict[str, str]] = []
    matrix_manifest = _json(matrix_dir / "matrix_manifest.json")
    protocol_path = matrix_dir / "protocol.json"
    protocol = _json(protocol_path)
    protocol_sha256 = _sha256(protocol_path)
    declared_protocol_sha = matrix_manifest.get("protocol_sha256")
    if declared_protocol_sha != protocol_sha256:
        _violation(
            violations,
            "PROTOCOL_SHA256_MISMATCH",
            f"declared={declared_protocol_sha}, actual={protocol_sha256}",
        )
    if protocol.get("protocol_id") != matrix_manifest.get("protocol_id"):
        _violation(violations, "PROTOCOL_ID_MISMATCH", "protocol IDs differ")

    planned = {
        str(_mapping(cell, name="planned cell").get("cell_id")): _mapping(
            cell, name="planned cell"
        )
        for cell in _array(matrix_manifest.get("planned_cells"), name="planned_cells")
    }
    _audit_matrix_protocol_stage(
        matrix_manifest,
        protocol,
        planned,
        violations=violations,
    )
    cells_dir = matrix_dir / "cells"
    observed = {path.name: path for path in cells_dir.iterdir() if path.is_dir()}
    if planned.keys() != observed.keys():
        _violation(
            violations,
            "MATRIX_CELL_SET_MISMATCH",
            (
                f"missing={sorted(planned.keys() - observed.keys())}, "
                f"extra={sorted(observed.keys() - planned.keys())}"
            ),
        )

    exact_predicates, mapper_key = _target_hole_keys()
    cell_reports: list[dict[str, object]] = []
    pair_data: dict[int, dict[str, tuple[object, object]]] = {}
    qd_splice_values: list[float] = []
    qd_mapper_qwen_ranks: list[int] = []
    qd_mapper_proposal_ranks: list[int] = []
    qd_predicate_qwen_ranks: list[int] = []
    qd_predicate_proposal_ranks: list[int] = []
    qd_exact_prefix_requests = 0
    aggregate = {
        "logical_score_requests": 0,
        "logical_candidates": 0,
        "provider_candidates": 0,
        "cache_hit_candidates": 0,
        "provider_scored_token_positions": 0,
        "cache_served_token_positions": 0,
        "ledger_token_positions": 0,
        "provider_score_requests": 0,
        "provider_invocations": 0,
        "provider_invocation_failures": 0,
        "http_requests": 0,
        "provider_http_failures": 0,
    }
    config = load_experiment_config(_experiment_path(protocol))
    options = _options(protocol)
    support_math, support, exact_traces, materialized_states = _materialize_support_math(
        config,
        options,
        bounded_square_target(),
    )
    runtime_support = _runtime_support(config, options)
    if len(runtime_support.states) != support.support_states:
        _violation(
            violations,
            "RUNTIME_FACTORIZED_SUPPORT_SIZE_MISMATCH",
            f"runtime={len(runtime_support.states)}, factorized={support.support_states}",
        )

    for cell_id in sorted(planned):
        cell_dir = observed.get(cell_id)
        if cell_dir is None:
            continue
        cell_document = _json(cell_dir / "cell.json")
        cell_plan = _mapping(cell_document.get("cell"), name="cell plan")
        if dict(cell_plan) != dict(planned[cell_id]):
            _violation(
                violations,
                "CELL_PLAN_MISMATCH",
                "cell.json plan differs from matrix manifest",
                cell_id=cell_id,
            )
        if cell_document.get("protocol_sha256") != protocol_sha256:
            _violation(
                violations,
                "CELL_PROTOCOL_SHA256_MISMATCH",
                "cell protocol digest differs",
                cell_id=cell_id,
            )
        if cell_document.get("status") != "completed" or cell_document.get("exit_code") != 0:
            _violation(
                violations,
                "CELL_NOT_COMPLETED",
                f"status={cell_document.get('status')}, exit={cell_document.get('exit_code')}",
                cell_id=cell_id,
            )
        artifact_relative = cell_document.get("core_artifact")
        if not isinstance(artifact_relative, str):
            _violation(
                violations,
                "CORE_ARTIFACT_MISSING",
                "cell lacks core_artifact",
                cell_id=cell_id,
            )
            continue
        artifact = cell_dir / artifact_relative
        core_manifest = _json(artifact / "manifest.json")
        envelope = _json(artifact / "result.json")
        result = _mapping(envelope.get("result"), name="result envelope.result")
        if (
            envelope.get("run_id") != core_manifest.get("run_id")
            or envelope.get("status") != "completed"
            or core_manifest.get("status") != "completed"
        ):
            _violation(
                violations,
                "CORE_RUN_ID_OR_STATUS_MISMATCH",
                "core result/manifest run identity or status differs",
                cell_id=cell_id,
            )
        telemetry, telemetry_violations = reconcile_telemetry(
            result,
            core_manifest,
            cell_id=cell_id,
        )
        violations.extend(telemetry_violations)
        _audit_finite_reference(
            result,
            support_math,
            cell_id=cell_id,
            violations=violations,
        )
        for key in aggregate:
            aggregate[key] += _integer(telemetry.get(key), name=f"telemetry.{key}")

        arm = str(cell_plan.get("arm"))
        seed = _integer(cell_plan.get("seed"), name="cell seed")
        ledgers = tuple(
            _mapping(value, name="score ledger")
            for value in _array(result.get("score_ledger"), name="score_ledger")
        )
        _audit_result_mixes(
            result,
            ledgers,
            protocol=protocol,
            arm=arm,
            cell_id=cell_id,
            violations=violations,
        )
        _audit_protocol_configuration(
            result,
            envelope,
            cell_document,
            core_manifest,
            matrix_manifest,
            protocol,
            cell_plan,
            ledgers,
            config,
            cell_id=cell_id,
            violations=violations,
        )
        qd_report: dict[str, object] | None = None
        if arm == "QD":
            qd_report, ledger_violations = audit_qd_target_ledgers(
                ledgers,
                exact_predicate_keys=exact_predicates,
                mapper_key=mapper_key,
                cell_id=cell_id,
            )
            violations.extend(ledger_violations)
            qd_exact_prefix_requests += _integer(
                qd_report.get("exact_prefix_mapper_requests"),
                name="exact_prefix_mapper_requests",
            )
            qd_mapper_qwen_ranks.extend(cast(list[int], qd_report["mapper_qwen_ranks"]))
            qd_mapper_proposal_ranks.extend(
                cast(list[int], qd_report["mapper_proposal_ranks"])
            )
            qd_predicate_qwen_ranks.extend(
                cast(list[int], qd_report["predicate_qwen_ranks"])
            )
            qd_predicate_proposal_ranks.extend(
                cast(list[int], qd_report["predicate_proposal_ranks"])
            )
            splice = cast(Mapping[str, object], qd_report["splice_proxy"])
            observations = int(cast(int, splice["observations"]))
            if observations:
                # Preserve exact per-cell weighting when aggregating the 17 observed branches.
                per_cell_mean = cast(float, splice["mean"])
                qd_splice_values.extend([per_cell_mean] * observations)
        else:
            for ledger in ledgers:
                _replay_ledger(ledger, cell_id=cell_id, violations=violations)

        _audit_catalog_and_deduction_guides(
            ledgers,
            support,
            config,
            options,
            cell_id=cell_id,
            violations=violations,
        )

        particles = _array(result.get("final_particles"), name="final_particles")
        particle_records = tuple(
            _mapping(particle, name="particle") for particle in particles
        )
        _audit_score_prompt_provenance(
            particle_records,
            ledgers,
            support,
            runtime_support,
            config,
            cell_id=cell_id,
            violations=violations,
        )
        _audit_ledger_particle_paths(
            particle_records,
            ledgers,
            support,
            allowed_integer_constants=config.spec.integer_constants,
            cell_id=cell_id,
            violations=violations,
        )
        _audit_final_particle_math(
            particle_records,
            materialized_states,
            runtime_support,
            cell_id=cell_id,
            violations=violations,
        )
        _audit_sampled_best(
            result,
            particle_records,
            materialized_states,
            runtime_support,
            support_math,
            cell_id=cell_id,
            violations=violations,
        )
        trace_signature = [
            _particle_path_signature(particle) for particle in particle_records
        ]
        pair_data.setdefault(seed, {})[arm] = (
            trace_signature,
            _selection_signature(ledgers),
        )
        best, evaluated_programs, result_shape = _result_search_summary(result)
        cell_reports.append(
            {
                "cell_id": cell_id,
                "arm": arm,
                "seed": seed,
                "support_states": result.get("support_states"),
                "evaluated_programs": evaluated_programs,
                "result_shape": result_shape,
                "training_exact": best.get("exact_program"),
                "exact_scope": result_shape,
                "final_exact_particles": sum(
                    bool(particle.get("exact_program")) for particle in particle_records
                ),
                "best_loss": best.get("total_loss"),
                "telemetry": telemetry,
                "qd_target": qd_report,
            }
        )
        del envelope

    pairs = []
    for seed, arms in sorted(pair_data.items()):
        arms_present = "D" in arms and "QD" in arms
        same_traces = arms_present and arms["D"][0] == arms["QD"][0]
        same_selections = arms_present and arms["D"][1] == arms["QD"][1]
        if not arms_present:
            _violation(
                violations,
                "PAIRED_ARM_MISSING",
                f"seed={seed}, arms={sorted(arms)}",
            )
        pairs.append(
            {
                "seed": seed,
                "paired_arms_present": arms_present,
                "same_ancestor_and_final_traces": same_traces,
                "same_selected_indices": same_selections,
            }
        )

    declared_supports = {int(cast(int, cell["support_states"])) for cell in cell_reports}
    if declared_supports != {support.support_states}:
        _violation(
            violations,
            "MATERIALIZED_SUPPORT_MISMATCH",
            f"cells={sorted(declared_supports)}, materialized={support.support_states}",
        )
    if not math.isclose(
        cast(float, support_math["prior_total"]), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        _violation(
            violations,
            "PRIOR_NOT_NORMALIZED",
            f"sum={support_math['prior_total']}",
        )
    d_probability = _d_guided_probability(config, options, support, exact_traces)

    qd_splice_mean = (
        None if not qd_splice_values else math.fsum(qd_splice_values) / len(qd_splice_values)
    )
    qd_summary: dict[str, object] = {
        "exact_proposal_probability": {
            "status": NOT_IDENTIFIED,
            "value": None,
            "reason": "no exact-prefix mapper request exists in any QD cell",
        },
        "exact_prefix_mapper_requests": qd_exact_prefix_requests,
        "splice_proxy": {
            "status": NONIDENTIFYING_SPLICE_PROXY,
            "eligible_for_discovery_math": False,
            "observations": len(qd_splice_values),
            "mean": qd_splice_mean,
            "minimum_cell_weighted_mean": (
                None if not qd_splice_values else min(qd_splice_values)
            ),
            "maximum_cell_weighted_mean": (
                None if not qd_splice_values else max(qd_splice_values)
            ),
        },
        "discovery_probability_20_draws": {
            "status": REFUSED_NOT_IDENTIFIED,
            "value": None,
        },
        "draws_for_50_percent": {"status": REFUSED_NOT_IDENTIFIED, "value": None},
        "mapper_qwen_rank_range": (
            None
            if not qd_mapper_qwen_ranks
            else [min(qd_mapper_qwen_ranks), max(qd_mapper_qwen_ranks)]
        ),
        "mapper_proposal_rank_range": (
            None
            if not qd_mapper_proposal_ranks
            else [min(qd_mapper_proposal_ranks), max(qd_mapper_proposal_ranks)]
        ),
        "predicate_qwen_rank_range": (
            None
            if not qd_predicate_qwen_ranks
            else [min(qd_predicate_qwen_ranks), max(qd_predicate_qwen_ranks)]
        ),
        "predicate_proposal_rank_range": (
            None
            if not qd_predicate_proposal_ranks
            else [
                min(qd_predicate_proposal_ranks),
                max(qd_predicate_proposal_ranks),
            ]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "deduction-stress-math",
        "audit_source_sha256": _sha256(Path(__file__).resolve()),
        "matrix_dir": str(matrix_dir),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "invariants_passed": not violations,
        "violations": violations,
        "matrix": {
            "planned_cells": len(planned),
            "observed_cells": len(observed),
            "paired_draws": pairs,
        },
        "telemetry": {
            **aggregate,
            "local_catalog_logical_candidates": sum(
                _integer(
                    cast(Mapping[str, object], cell["telemetry"]).get(
                        "logical_candidates"
                    ),
                    name="logical_candidates",
                )
                for cell in cell_reports
                if cell["arm"] == "D"
            ),
            "provider_backed_logical_candidates": sum(
                _integer(
                    cast(Mapping[str, object], cell["telemetry"]).get(
                        "logical_candidates"
                    ),
                    name="logical_candidates",
                )
                for cell in cell_reports
                if cell["arm"] == "QD"
            ),
            "reconciliation_equation": (
                "ledger_token_positions = provider_scored_token_positions + "
                "cache_served_token_positions"
            ),
            "http_requests_are_not_logical_score_requests": True,
            "http_request_seconds_are_not_wall_seconds": True,
        },
        "target_math": support_math,
        "d_guided_exact_proposal": d_probability,
        "qd_guided_exact_proposal": qd_summary,
        "cells": cell_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero only when an audit invariant is violated",
    )
    args = parser.parse_args(argv)
    report = audit_matrix(args.matrix)
    if args.output is not None:
        write_json_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 2 if args.strict and report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

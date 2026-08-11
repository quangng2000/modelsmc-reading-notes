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
from modelsmc_pbe.search.importance.lazy_records import (
    ConstructionTrace,
    FactorizedFamily,
    FactorizedImportanceSupport,
)
from modelsmc_pbe.search.importance.proposal_distribution import candidate_distribution
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
    return cast(list[Any], value)


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
    ledger_candidates = 0
    ledger_token_positions = 0
    for ledger in ledgers:
        record = _mapping(ledger, name="score ledger")
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
    if not cache or cache.get("mode") == "off":
        return (
            {
                "logical_score_requests": len(ledgers),
                "logical_candidates": scored_candidates,
                "provider_score_requests": 0,
                "http_requests": 0,
                "provider_candidates": 0,
                "cache_hit_candidates": 0,
                "provider_scored_token_positions": 0,
                "cache_served_token_positions": 0,
                "ledger_token_positions": ledger_token_positions,
            },
            violations,
        )

    lookup_requests = _integer(cache.get("lookup_requests"), name="lookup_requests")
    hit_requests = _integer(cache.get("hit_requests"), name="hit_requests")
    miss_requests = _integer(cache.get("miss_requests"), name="miss_requests")
    provider_score_requests = _integer(
        cache.get("provider_score_requests"), name="provider_score_requests"
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
    event_provider_tokens = _integer(
        provider.get("scored_token_positions"), name="provider scored_token_positions"
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
    )
    for valid, code, detail in equations:
        if not valid:
            _violation(violations, code, detail, cell_id=cell_id)

    return (
        {
            "logical_score_requests": lookup_requests,
            "logical_candidates": lookup_candidates,
            "provider_score_requests": provider_score_requests,
            "http_requests": http_requests,
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
    dict[str, object], FactorizedImportanceSupport, tuple[ConstructionTrace, ...]
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
                    - float(config.smc.loss_scale) * score.total_loss,
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
        / threshold_competitor["loss"],
    )
    prior_total = math.fsum(math.exp(state["log_prior"]) for state in all_states)
    exact_prior_mass = math.fsum(math.exp(state["log_prior"]) for state in exact)
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
    return report, support, exact_traces


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
        deduction_mix=options.deduction_mix,
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
                deduction_mix=options.deduction_mix,
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
    for replayed_value, raw_candidate in zip(replayed, candidates, strict=True):
        candidate = _mapping(raw_candidate, name="ledger candidate")
        deduction = _number(
            candidate.get("deduction_probability"), name="deduction_probability"
        )
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
    if not math.isclose(math.fsum(proposal), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        _violation(
            violations,
            "PROPOSAL_NOT_NORMALIZED",
            f"sum={math.fsum(proposal)}",
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


def _options(protocol: Mapping[str, Any]) -> ImportanceSMCOptions:
    caps = _mapping(protocol.get("caps"), name="protocol.caps")
    shared = _mapping(protocol.get("shared_arguments"), name="protocol.shared_arguments")
    return ImportanceSMCOptions(
        hole_max_cost=_integer(caps.get("hole_max_cost"), name="hole_max_cost"),
        hole_state_limit=_integer(caps.get("hole_state_limit"), name="hole_state_limit"),
        support_limit=_integer(caps.get("support_limit"), name="support_limit"),
        proposal_temperature=_number(shared.get("temperature"), name="temperature"),
        proposal_epsilon=_number(shared.get("proposal_epsilon"), name="proposal_epsilon"),
        deduction_mix=0.75,
        deduction_strength=_number(
            shared.get("deduction_strength"), name="deduction_strength"
        ),
        beta_max=_number(shared.get("beta_max"), name="beta_max"),
        max_scored_candidates=_integer(
            caps.get("max_scored_candidates"), name="max_scored_candidates"
        ),
        multi_family=True,
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
                    selection.get("slot"),
                    selection.get("selected_index"),
                ]
            )
    return signature


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
        "http_requests": 0,
    }

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
        for key in aggregate:
            aggregate[key] += int(telemetry[key])

        arm = str(cell_plan.get("arm"))
        seed = _integer(cell_plan.get("seed"), name="cell seed")
        ledgers = tuple(
            _mapping(value, name="score ledger")
            for value in _array(result.get("score_ledger"), name="score_ledger")
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
            qd_exact_prefix_requests += int(qd_report["exact_prefix_mapper_requests"])
            qd_mapper_qwen_ranks.extend(cast(list[int], qd_report["mapper_qwen_ranks"]))
            qd_mapper_proposal_ranks.extend(
                cast(list[int], qd_report["mapper_proposal_ranks"])
            )
            splice = cast(Mapping[str, object], qd_report["splice_proxy"])
            observations = int(cast(int, splice["observations"]))
            if observations:
                # Preserve exact per-cell weighting when aggregating the 17 observed branches.
                per_cell_mean = cast(float, splice["mean"])
                qd_splice_values.extend([per_cell_mean] * observations)

        particles = _array(result.get("final_particles"), name="final_particles")
        trace_signature = [
            [
                _mapping(particle, name="particle").get("trace"),
                _mapping(particle, name="particle").get("ancestor_trace"),
            ]
            for particle in particles
        ]
        pair_data.setdefault(seed, {})[arm] = (
            trace_signature,
            _selection_signature(ledgers),
        )
        search = _mapping(result.get("search"), name="result.search")
        best = _mapping(result.get("best_visited"), name="result.best_visited")
        cell_reports.append(
            {
                "cell_id": cell_id,
                "arm": arm,
                "seed": seed,
                "support_states": result.get("support_states"),
                "evaluated_programs": search.get("evaluated_programs"),
                "training_exact": best.get("exact_program"),
                "best_loss": best.get("total_loss"),
                "telemetry": telemetry,
                "qd_target": qd_report,
            }
        )
        del envelope

    pairs = []
    for seed, arms in sorted(pair_data.items()):
        same_traces = "D" in arms and "QD" in arms and arms["D"][0] == arms["QD"][0]
        same_selections = "D" in arms and "QD" in arms and arms["D"][1] == arms["QD"][1]
        if not same_traces or not same_selections:
            _violation(
                violations,
                "PAIRED_DRAW_IDENTITY_MISMATCH",
                f"seed={seed}, traces={same_traces}, selections={same_selections}",
            )
        pairs.append(
            {
                "seed": seed,
                "same_ancestor_and_final_traces": same_traces,
                "same_selected_indices": same_selections,
            }
        )

    config = load_experiment_config(_experiment_path(protocol))
    options = _options(protocol)
    support_math, support, exact_traces = _materialize_support_math(
        config,
        options,
        bounded_square_target(),
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
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "deduction-stress-math",
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
                int(cast(Mapping[str, object], cell["telemetry"])["logical_candidates"])
                for cell in cell_reports
                if cell["arm"] == "D"
            ),
            "provider_backed_logical_candidates": sum(
                int(cast(Mapping[str, object], cell["telemetry"])["logical_candidates"])
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

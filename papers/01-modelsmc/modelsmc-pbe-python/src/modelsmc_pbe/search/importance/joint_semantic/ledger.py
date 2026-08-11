"""Compact replay evidence for joint-semantic proposal scores and draws."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from modelsmc_pbe.proposals.labels import CompatibilityScore


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticTraceIdentity:
    """Portable identity for one factorized construction trace."""

    hypothesis_index: int
    filling_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SemanticBoundaryLedger:
    """Hashed shared path plus the two final label-token scores."""

    mapping: str
    label_a_candidate_sha256: str
    label_b_candidate_sha256: str
    label_a_token_ids_sha256: str
    label_b_token_ids_sha256: str
    label_a_token_logprobs_sha256: str
    label_b_token_logprobs_sha256: str
    label_a_prefix_token_ids_sha256: str
    label_b_prefix_token_ids_sha256: str
    label_a_prefix_token_logprobs_sha256: str
    label_b_prefix_token_logprobs_sha256: str
    shared_token_count: int
    shared_token_ids_sha256: str
    shared_token_logprobs_sha256: str
    label_a_token_id: int
    label_b_token_id: int
    label_a_logprob: float
    label_b_logprob: float


@dataclass(frozen=True, slots=True)
class SemanticProgramScoreLedger:
    """One symmetrized compatibility score without duplicating full token paths."""

    program_sha256: str
    compatibility_log_odds: float
    template_version: str
    prompt_sha256: str
    source: str
    model: str
    score_semantics: str
    model_revision: str | None
    tokenizer_revision: str | None
    score_origin: str
    cache_key_sha256: str | None
    cache_hit: bool | None
    boundaries: tuple[SemanticBoundaryLedger, ...]


@dataclass(frozen=True, slots=True)
class SemanticTraceProbabilityLedger:
    """Semantic and defensive masses for one trace in the scored slate."""

    trace: SemanticTraceIdentity
    program_sha256: str
    log_prior: float
    compatibility_log_odds: float
    log_q_semantic: float | None
    log_q_proposal: float


@dataclass(frozen=True, slots=True)
class SemanticSelectionLedger:
    """One sequential joint draw and its telescoping conditional factors."""

    stage: int
    beta: float
    slot: int
    ancestor: SemanticTraceIdentity
    selected: SemanticTraceIdentity
    family_log_probability: float
    hole_log_probabilities: tuple[float, ...]
    selected_log_prior: float
    selected_log_q_semantic: float | None
    log_q_proposal: float


@dataclass(frozen=True, slots=True)
class SemanticProposalLedger:
    """Derived proposal arithmetic plus hashes committing to raw score evidence."""

    formula: str
    slate_selection: str
    slate_seed: int
    support_states: int
    requested_slate_size: int | None
    slate_traces: int
    unique_programs: int
    raw_candidate_count: int
    proposal_epsilon: float
    semantic_scale: float
    template_version: str
    prompt_sha256: str
    slate_sha256: str
    program_scores: tuple[SemanticProgramScoreLedger, ...]
    trace_probabilities: tuple[SemanticTraceProbabilityLedger, ...]
    selections: tuple[SemanticSelectionLedger, ...]

    def __post_init__(self) -> None:
        if self.slate_traces != len(self.trace_probabilities):
            raise ValueError("semantic trace ledger count disagrees with slate size")
        if self.unique_programs != len(self.program_scores):
            raise ValueError("semantic program ledger count disagrees with unique programs")
        if self.raw_candidate_count != 4 * self.unique_programs:
            raise ValueError("joint-semantic scoring requires four raw paths per program")
        if any(
            not math.isfinite(value)
            for trace in self.trace_probabilities
            for value in (
                trace.log_prior,
                trace.compatibility_log_odds,
                trace.log_q_proposal,
            )
        ):
            raise ValueError("semantic trace ledger probabilities must be finite")
        from .replay import validate_semantic_proposal_ledger

        validate_semantic_proposal_ledger(self)


def validate_final_semantic_selections(
    ledger: SemanticProposalLedger,
    *,
    iterations: int,
    final_traces: tuple[SemanticTraceIdentity, ...],
    final_ancestors: tuple[SemanticTraceIdentity, ...],
    final_log_q: tuple[float, ...],
) -> None:
    """Bind the complete selection ledger to the final particle population."""

    particles = len(final_traces)
    if iterations < 1 or particles < 1:
        raise ValueError("semantic final selection dimensions must be positive")
    if len(final_ancestors) != particles or len(final_log_q) != particles:
        raise ValueError("semantic final particle arrays must align")
    expected_keys = {
        (stage, slot)
        for stage in range(1, iterations + 1)
        for slot in range(particles)
    }
    actual = {(selection.stage, selection.slot): selection for selection in ledger.selections}
    if set(actual) != expected_keys:
        raise ValueError("semantic selection ledger is incomplete")
    for slot, (trace, ancestor, log_q) in enumerate(
        zip(final_traces, final_ancestors, final_log_q, strict=True)
    ):
        selection = actual[(iterations, slot)]
        if selection.selected != trace or selection.ancestor != ancestor:
            raise ValueError("semantic final selection path disagrees with particle")
        if not math.isclose(
            selection.log_q_proposal,
            log_q,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("semantic final selection density disagrees with particle")


def trace_identity(
    hypothesis_index: int,
    filling_indices: tuple[int, ...],
) -> SemanticTraceIdentity:
    """Build a portable trace identity without importing lazy record types."""

    return SemanticTraceIdentity(hypothesis_index, filling_indices)


def summarize_score(score: CompatibilityScore) -> SemanticProgramScoreLedger:
    """Reduce raw label paths to hashed boundary evidence and final logprobs."""

    provenance = score.raw_identity.provenance
    boundaries = tuple(
        SemanticBoundaryLedger(
            mapping=proof.mapping.value,
            label_a_candidate_sha256=score.paths[2 * index].candidate_sha256,
            label_b_candidate_sha256=score.paths[2 * index + 1].candidate_sha256,
            label_a_token_ids_sha256=score.paths[2 * index].token_ids_sha256,
            label_b_token_ids_sha256=score.paths[2 * index + 1].token_ids_sha256,
            label_a_token_logprobs_sha256=(
                score.paths[2 * index].token_logprobs_sha256
            ),
            label_b_token_logprobs_sha256=(
                score.paths[2 * index + 1].token_logprobs_sha256
            ),
            label_a_prefix_token_ids_sha256=(
                score.paths[2 * index].prefix_token_ids_sha256
            ),
            label_b_prefix_token_ids_sha256=(
                score.paths[2 * index + 1].prefix_token_ids_sha256
            ),
            label_a_prefix_token_logprobs_sha256=(
                score.paths[2 * index].prefix_token_logprobs_sha256
            ),
            label_b_prefix_token_logprobs_sha256=(
                score.paths[2 * index + 1].prefix_token_logprobs_sha256
            ),
            shared_token_count=proof.shared_token_count,
            shared_token_ids_sha256=proof.shared_token_ids_sha256,
            shared_token_logprobs_sha256=proof.shared_token_logprobs_sha256,
            label_a_token_id=proof.label_a_token_id,
            label_b_token_id=proof.label_b_token_id,
            label_a_logprob=proof.label_a_logprob,
            label_b_logprob=proof.label_b_logprob,
        )
        for index, proof in enumerate(score.boundary_proofs)
    )
    return SemanticProgramScoreLedger(
        program_sha256=score.program_sha256,
        compatibility_log_odds=score.compatibility_log_odds,
        template_version=score.template_version,
        prompt_sha256=score.prompt_sha256,
        source=score.raw_identity.source,
        model=score.raw_identity.model,
        score_semantics=score.raw_identity.semantics.value,
        model_revision=score.raw_identity.model_revision,
        tokenizer_revision=score.raw_identity.tokenizer_revision,
        score_origin="unspecified" if provenance is None else provenance.origin.value,
        cache_key_sha256=None if provenance is None else provenance.cache_key_sha256,
        cache_hit=None if provenance is None else provenance.cache_hit,
        boundaries=boundaries,
    )


def slate_digest(records: tuple[SemanticTraceProbabilityLedger, ...]) -> str:
    """Hash the complete ordered trace-to-program semantic law."""

    return _sha256(
        tuple(
            (
                item.trace.hypothesis_index,
                item.trace.filling_indices,
                item.program_sha256,
                item.log_prior,
                item.compatibility_log_odds,
                item.log_q_semantic,
                item.log_q_proposal,
            )
            for item in records
        )
    )

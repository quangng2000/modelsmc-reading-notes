"""Independent arithmetic replay for compact joint-semantic ledgers."""

from __future__ import annotations

import math

from modelsmc_pbe.proposals.candidate_scoring import CandidateLogprobSemantics

from .ledger import (
    SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION,
    SemanticBoundaryLedger,
    SemanticProgramScoreLedger,
    SemanticProposalLedger,
    SemanticTraceIdentity,
    SemanticTraceProbabilityLedger,
    slate_digest,
)
from .replay_math import (
    ABSOLUTE_TOLERANCE,
    close,
    defensive_log_probability,
    semantic_log_probabilities,
)

_CANONICAL_MAPPINGS = ("a-is-compatible", "b-is-compatible")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_boundary(boundary: SemanticBoundaryLedger) -> None:
    digests = (
        boundary.label_a_candidate_sha256,
        boundary.label_b_candidate_sha256,
        boundary.label_a_token_ids_sha256,
        boundary.label_b_token_ids_sha256,
        boundary.label_a_token_logprobs_sha256,
        boundary.label_b_token_logprobs_sha256,
        boundary.label_a_prefix_token_ids_sha256,
        boundary.label_b_prefix_token_ids_sha256,
        boundary.label_a_prefix_token_logprobs_sha256,
        boundary.label_b_prefix_token_logprobs_sha256,
        boundary.shared_token_ids_sha256,
    )
    if not all(_is_sha256(digest) for digest in digests):
        raise ValueError("semantic boundary evidence must use lowercase SHA-256")
    if boundary.label_a_candidate_sha256 == boundary.label_b_candidate_sha256:
        raise ValueError("semantic A/B candidates must have distinct identities")
    if boundary.shared_token_count < 0:
        raise ValueError("semantic boundary shared-token count must be nonnegative")
    if (
        boundary.label_a_prefix_token_ids_sha256 != boundary.shared_token_ids_sha256
        or boundary.label_b_prefix_token_ids_sha256 != boundary.shared_token_ids_sha256
    ):
        raise ValueError("semantic boundary prefix token IDs disagree")
    if (
        not math.isfinite(boundary.max_abs_prefix_logprob_delta)
        or boundary.max_abs_prefix_logprob_delta < 0.0
    ):
        raise ValueError("semantic prefix logprob delta must be finite and nonnegative")
    if (
        boundary.label_a_prefix_token_logprobs_sha256
        == boundary.label_b_prefix_token_logprobs_sha256
        and boundary.max_abs_prefix_logprob_delta != 0.0
    ):
        raise ValueError("identical semantic prefix logprob commitments require zero delta")
    if boundary.label_a_token_id < 0 or boundary.label_b_token_id < 0:
        raise ValueError("semantic label token IDs must be nonnegative")
    if boundary.label_a_token_id == boundary.label_b_token_id:
        raise ValueError("semantic A/B labels must have distinct token IDs")
    if any(
        not math.isfinite(value) or value > 0.0
        for value in (boundary.label_a_logprob, boundary.label_b_logprob)
    ):
        raise ValueError("semantic final-label logprobs must be finite and nonpositive")


def _validate_program_scores(
    ledger: SemanticProposalLedger,
) -> dict[str, SemanticProgramScoreLedger]:
    programs: dict[str, SemanticProgramScoreLedger] = {}
    for program in ledger.program_scores:
        if not _is_sha256(program.program_sha256):
            raise ValueError("semantic program identity must use lowercase SHA-256")
        if program.program_sha256 in programs:
            raise ValueError("semantic program score identities must be unique")
        if (
            program.template_version != ledger.template_version
            or program.prompt_sha256 != ledger.prompt_sha256
        ):
            raise ValueError("semantic program score prompt identity disagrees with ledger")
        if program.score_semantics != CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT:
            raise ValueError("semantic score must retain teacher-forced path semantics")
        if not program.source or not program.model:
            raise ValueError("semantic program score provider identity must not be empty")
        if program.cache_key_sha256 is not None and not _is_sha256(program.cache_key_sha256):
            raise ValueError("semantic cache identity must use lowercase SHA-256")
        provenance = (
            program.score_origin,
            program.cache_key_sha256,
            program.cache_hit,
        )
        if provenance[0] == "cache" and (provenance[1] is None or provenance[2] is not True):
            raise ValueError("cache-origin semantic scores require a hit and cache key")
        if provenance[0] in {"synthetic", "unspecified"} and provenance[1:] != (
            None,
            None,
        ):
            raise ValueError("uncached semantic score origins cannot have cache evidence")
        if provenance[0] == "provider" and (
            (provenance[1] is None and provenance[2] is not None)
            or (provenance[1] is not None and provenance[2] is not False)
        ):
            raise ValueError("provider semantic cache disposition is inconsistent")
        if provenance[0] not in {"cache", "provider", "synthetic", "unspecified"}:
            raise ValueError("unknown semantic score origin")
        if tuple(boundary.mapping for boundary in program.boundaries) != _CANONICAL_MAPPINGS:
            raise ValueError("semantic score requires both canonical swapped mappings")
        for boundary in program.boundaries:
            _validate_boundary(boundary)
        plus, minus = program.boundaries
        reconstructed = 0.5 * (
            plus.label_a_logprob
            - plus.label_b_logprob
            + minus.label_b_logprob
            - minus.label_a_logprob
        )
        if not close(program.compatibility_log_score, reconstructed):
            raise ValueError("semantic program score disagrees with label evidence")
        programs[program.program_sha256] = program
    return programs


def _validate_trace_probabilities(
    ledger: SemanticProposalLedger,
    programs: dict[str, SemanticProgramScoreLedger],
) -> dict[SemanticTraceIdentity, SemanticTraceProbabilityLedger]:
    expected_q_a = semantic_log_probabilities(
        ledger.trace_probabilities,
        semantic_scale=ledger.semantic_scale,
    )
    traces: dict[SemanticTraceIdentity, SemanticTraceProbabilityLedger] = {}
    for record, expected_semantic in zip(
        ledger.trace_probabilities,
        expected_q_a,
        strict=True,
    ):
        if record.trace in traces:
            raise ValueError("semantic trace identities must be unique")
        program = programs.get(record.program_sha256)
        if program is None:
            raise ValueError("semantic trace references an unknown program score")
        if not close(record.compatibility_log_score, program.compatibility_log_score):
            raise ValueError("semantic trace score disagrees with its program evidence")
        stored_semantic = -math.inf if record.log_q_semantic is None else record.log_q_semantic
        if not close(stored_semantic, expected_semantic):
            raise ValueError("semantic component probability failed replay")
        expected_proposal = defensive_log_probability(
            epsilon=ledger.proposal_epsilon,
            log_prior=record.log_prior,
            log_q_semantic=record.log_q_semantic,
        )
        if not close(record.log_q_proposal, expected_proposal):
            raise ValueError("defensive semantic probability failed replay")
        if record.log_prior > ABSOLUTE_TOLERANCE or record.log_q_proposal > ABSOLUTE_TOLERANCE:
            raise ValueError("semantic trace prior and proposal masses cannot exceed one")
        traces[record.trace] = record
    prior_mass = math.fsum(math.exp(record.log_prior) for record in traces.values())
    proposal_mass = math.fsum(math.exp(record.log_q_proposal) for record in traces.values())
    if prior_mass > 1.0 + ABSOLUTE_TOLERANCE or proposal_mass > 1.0 + ABSOLUTE_TOLERANCE:
        raise ValueError("semantic slate prior or proposal mass exceeds one")
    if ledger.slate_traces == ledger.support_states and (
        not close(prior_mass, 1.0) or not close(proposal_mass, 1.0)
    ):
        raise ValueError("full semantic slate prior and proposal masses must normalize")
    if slate_digest(ledger.trace_probabilities) != ledger.slate_sha256:
        raise ValueError("semantic slate digest failed replay")
    if set(programs) != {record.program_sha256 for record in ledger.trace_probabilities}:
        raise ValueError("semantic program and trace score inventories disagree")
    return traces


def _validate_selections(
    ledger: SemanticProposalLedger,
    traces: dict[SemanticTraceIdentity, SemanticTraceProbabilityLedger],
) -> None:
    selection_keys: set[tuple[int, int]] = set()
    for selection in ledger.selections:
        key = (selection.stage, selection.slot)
        if key in selection_keys:
            raise ValueError("semantic selection stage/slot identities must be unique")
        selection_keys.add(key)
        if selection.stage < 1 or selection.slot < 0 or not math.isfinite(selection.beta):
            raise ValueError("semantic selection stage, slot, and beta must be valid")
        factors = (selection.family_log_probability, *selection.hole_log_probabilities)
        if len(selection.hole_log_probabilities) != len(selection.selected.filling_indices):
            raise ValueError("semantic selection requires one factor per selected hole")
        if any(not math.isfinite(value) or value > ABSOLUTE_TOLERANCE for value in factors):
            raise ValueError("semantic selection conditionals must be finite and nonpositive")
        sequential = math.fsum(factors)
        if not close(sequential, selection.log_q_proposal):
            raise ValueError("semantic selection conditionals failed to telescope")
        if (
            not math.isfinite(selection.selected_log_prior)
            or selection.selected_log_prior > ABSOLUTE_TOLERANCE
            or selection.log_q_proposal > ABSOLUTE_TOLERANCE
        ):
            raise ValueError("semantic selection prior must be finite")
        trace_record = traces.get(selection.selected)
        if trace_record is None:
            if selection.selected_log_q_semantic is not None:
                raise ValueError("outside-slate selection cannot have semantic component mass")
        else:
            if not close(selection.selected_log_prior, trace_record.log_prior):
                raise ValueError("semantic selection prior disagrees with slate evidence")
            stored = selection.selected_log_q_semantic
            if stored is None or trace_record.log_q_semantic is None:
                if stored != trace_record.log_q_semantic:
                    raise ValueError("semantic selection component mass disagrees with slate")
            elif not close(stored, trace_record.log_q_semantic):
                raise ValueError("semantic selection component mass disagrees with slate")
        direct = defensive_log_probability(
            epsilon=ledger.proposal_epsilon,
            log_prior=selection.selected_log_prior,
            log_q_semantic=selection.selected_log_q_semantic,
        )
        if not close(selection.log_q_proposal, direct):
            raise ValueError("selected defensive semantic probability failed replay")


def validate_semantic_proposal_ledger(ledger: SemanticProposalLedger) -> None:
    """Recompute every derived semantic score, normalized mass, and selected density."""

    if ledger.schema_version != SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION:
        raise ValueError("unknown joint-semantic proposal ledger schema")
    if ledger.formula != "q=epsilon*pi+(1-epsilon)*normalize_slate(pi*exp(eta*a_llm))":
        raise ValueError("unknown joint-semantic proposal formula")
    if ledger.slate_selection not in {"full-bounded-support", "seeded-sha256-rank"}:
        raise ValueError("unknown semantic slate selection method")
    if ledger.support_states < 1 or not 1 <= ledger.slate_traces <= ledger.support_states:
        raise ValueError("semantic support and slate counts are invalid")
    if ledger.requested_slate_size is None:
        if ledger.slate_selection != "full-bounded-support":
            raise ValueError("full semantic slate must declare full-support selection")
    elif ledger.requested_slate_size < 1 or ledger.slate_traces != min(
        ledger.requested_slate_size,
        ledger.support_states,
    ):
        raise ValueError("semantic requested slate size disagrees with selected traces")
    if not 0.0 < ledger.proposal_epsilon <= 1.0:
        raise ValueError("semantic proposal epsilon must be in (0, 1]")
    if not math.isfinite(ledger.semantic_scale) or ledger.semantic_scale < 0.0:
        raise ValueError("semantic scale must be finite and nonnegative")
    if not ledger.template_version or not _is_sha256(ledger.prompt_sha256):
        raise ValueError("semantic prompt identity is invalid")
    if not _is_sha256(ledger.slate_sha256):
        raise ValueError("semantic slate identity is invalid")
    programs = _validate_program_scores(ledger)
    traces = _validate_trace_probabilities(ledger, programs)
    _validate_selections(ledger, traces)

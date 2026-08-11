"""Hole-level waves for the finite guided construction proposal."""

from __future__ import annotations

import hashlib
import math

from modelsmc_pbe.proposals import CandidateScoreBatch, CandidateScoreRequest
from modelsmc_pbe.smc import categorical_sample

from ..prompts import hole_prompt_prefix, proposal_hole
from ..trie import GroupedHoleChoice, grouped_hole_choices
from .models import ProposalPath
from .runtime import GuidedProposalRuntime


async def complete_paths(
    runtime: GuidedProposalRuntime,
    paths: tuple[ProposalPath, ...],
    *,
    stage: int,
    beta: float,
) -> tuple[ProposalPath, ...]:
    maximum_holes = max(len(path.family.hypothesis.holes) for path in paths)
    for hole_index in range(maximum_holes):
        active = tuple(path for path in paths if hole_index < len(path.family.hypothesis.holes))
        requests = [
            _request(runtime, path, hole_index=hole_index, stage=stage, beta=beta)
            for path in active
        ]
        batches = await runtime.score_requests(
            requests,
            stage=stage,
            wave=f"hole-{hole_index}",
            event_scope="importance.proposal.scoring",
            event_message="finite-candidate scoring wave",
            hole_index=hole_index,
        )
        for path, request, batch in zip(active, requests, batches, strict=True):
            _advance(
                runtime,
                path,
                hole_index=hole_index,
                stage=stage,
                beta=beta,
                request=request,
                batch=batch,
            )
    return paths


def _choices(
    runtime: GuidedProposalRuntime,
    path: ProposalPath,
    hole_index: int,
) -> tuple[GroupedHoleChoice, ...]:
    return grouped_hole_choices(
        runtime.support,
        path.family,
        path.compatible_state_indices,
        hole_index=hole_index,
    )


def _request(
    runtime: GuidedProposalRuntime,
    path: ProposalPath,
    *,
    hole_index: int,
    stage: int,
    beta: float,
) -> CandidateScoreRequest:
    choices = _choices(runtime, path, hole_index)
    hole = path.family.hypothesis.holes[hole_index]
    signature = runtime.config.spec.signature
    if signature is None:  # pragma: no cover
        raise ValueError("PBE specification has no signature")
    return CandidateScoreRequest(
        prompt_prefix=hole_prompt_prefix(
            config=runtime.config,
            report=path.family.deduction,
            hole=hole,
            ancestor=runtime.support.states[path.ancestor_state_index],
            previous_fillings=path.previous_fillings,
            stage=stage,
            beta=beta,
            candidate_count=len(choices),
        ),
        candidates=tuple(choice.key for choice in choices),
        hole=proposal_hole(hole, input_type=signature.input_type),
        integer_constants=tuple(runtime.config.spec.integer_constants),
        request_index=runtime.next_request_index(),
        max_depth=runtime.config.smc.max_depth,
        max_nodes=runtime.config.smc.max_nodes,
    )


def _advance(
    runtime: GuidedProposalRuntime,
    path: ProposalPath,
    *,
    hole_index: int,
    stage: int,
    beta: float,
    request: CandidateScoreRequest,
    batch: CandidateScoreBatch,
) -> None:
    choices = _choices(runtime, path, hole_index)
    deduction_mix = runtime.options.resolved_hole_deduction_mix
    distribution = runtime.candidate_probabilities(
        request,
        batch,
        groups=tuple(choice.state_indices for choice in choices),
        beta=beta,
        deduction_mix=deduction_mix,
    )
    probabilities = distribution.probabilities
    if path.target_state_index is None:
        selected = int(categorical_sample(probabilities, 1, generator=runtime.generator)[0].item())
    else:
        target = runtime.support.states[path.target_state_index]
        target_key = target.filling(path.family.hypothesis.holes[hole_index].name).key
        try:
            selected = request.candidates.index(target_key)
        except ValueError as error:  # pragma: no cover - support-trie invariant
            raise RuntimeError("target state is absent from its construction trie") from error
    probability = float(probabilities[selected].item())
    if not math.isfinite(probability) or probability <= 0.0:
        raise RuntimeError("smoothed finite proposal produced nonpositive mass")
    chosen = choices[selected].filling
    runtime.record_score_wave(
        stage=stage,
        beta=beta,
        wave=f"hole-{hole_index}",
        request=request,
        batch=batch,
        distribution=distribution,
        selected=selected,
        slot=path.slot,
        ancestor_state_index=path.ancestor_state_index,
        forced=path.target_state_index is not None,
        cloned=path.cloned,
        deduction_mix=deduction_mix,
    )
    runtime.event(
        "importance.proposal.hole_selected",
        message="finite typed hole filling selected",
        level="trace",
        stage=stage,
        slot=path.slot,
        ancestor_state_index=path.ancestor_state_index,
        family=path.family.hypothesis.kind.value,
        hole=chosen.hole_name,
        candidate=chosen.key,
        candidate_sha256=hashlib.sha256(chosen.key.encode("utf-8")).hexdigest(),
        probability=probability,
        qwen_probability=float(distribution.q_llm[selected].item()),
        deduction_probability=float(distribution.q_deduction[selected].item()),
        uniform_floor=runtime.options.proposal_epsilon / len(choices),
        deduction_mix=deduction_mix,
        deduction_examples=len(path.family.deduction.examples_for(chosen.hole_name)),
        forced=path.target_state_index is not None,
        cloned=path.cloned,
    )
    path.log_q_construct += math.log(probability)
    path.previous_fillings.append(chosen)
    path.compatible_state_indices = choices[selected].state_indices
    if not path.compatible_state_indices:  # pragma: no cover - trie invariant
        raise RuntimeError("proposal choice has no valid complete program")

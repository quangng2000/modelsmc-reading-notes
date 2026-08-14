"""Family-level wave for the finite guided construction proposal."""

from __future__ import annotations

import math

from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateScoreBatch,
    CandidateScoreRequest,
)
from modelsmc_pbe.smc import categorical_sample

from ..prompts import family_candidate, family_prompt_prefix
from ..records import FamilySupport
from ..trie import ordered_families
from .models import PathSeed, ProposalPath
from .runtime import GuidedProposalRuntime


async def select_families(
    runtime: GuidedProposalRuntime,
    seeds: tuple[PathSeed, ...],
    *,
    stage: int,
    beta: float,
) -> tuple[ProposalPath, ...]:
    if len(runtime.support.families) == 1:
        family = runtime.support.families[0]
        paths: list[ProposalPath] = []
        for seed in seeds:
            if seed.target_state_index is not None:
                target = runtime.support.states[seed.target_state_index]
                if target.hypothesis_index != family.hypothesis_index:
                    raise RuntimeError("target state is outside the sole viable family")
            paths.append(
                ProposalPath(
                    slot=seed.slot,
                    ancestor_state_index=seed.ancestor_state_index,
                    family=family,
                    compatible_state_indices=family.state_indices,
                    cloned=seed.cloned,
                    target_state_index=seed.target_state_index,
                )
            )
        runtime.event(
            "importance.proposal.family_scoring.skipped",
            message="one viable skeleton family has categorical probability one",
            level="debug",
            stage=stage,
            paths=len(paths),
            family=family.hypothesis.kind.value,
        )
        return tuple(paths)

    choices = ordered_families(runtime.support)
    requests = [_family_request(runtime, seed, choices, stage=stage, beta=beta) for seed in seeds]
    batches = await runtime.score_requests(
        requests,
        stage=stage,
        wave="family",
        event_scope="importance.proposal.family_scoring",
        event_message="finite skeleton-family scoring wave",
    )
    return tuple(
        _select_family(
            runtime,
            seed,
            choices,
            request,
            batch,
            stage=stage,
            beta=beta,
        )
        for seed, request, batch in zip(seeds, requests, batches, strict=True)
    )


def _family_request(
    runtime: GuidedProposalRuntime,
    seed: PathSeed,
    choices: tuple[FamilySupport, ...],
    *,
    stage: int,
    beta: float,
) -> CandidateScoreRequest:
    return CandidateScoreRequest(
        prompt_prefix=family_prompt_prefix(
            config=runtime.config,
            families=choices,
            ancestor=runtime.support.states[seed.ancestor_state_index],
            stage=stage,
            beta=beta,
        ),
        candidates=tuple(family_candidate(family) for family in choices),
        hole=None,
        integer_constants=tuple(runtime.config.spec.integer_constants),
        request_index=runtime.next_request_index(),
        max_depth=runtime.config.smc.max_depth,
        max_nodes=runtime.config.smc.max_nodes,
        candidate_kind=CandidateKind.SKELETON,
    )


def _select_family(
    runtime: GuidedProposalRuntime,
    seed: PathSeed,
    choices: tuple[FamilySupport, ...],
    request: CandidateScoreRequest,
    batch: CandidateScoreBatch,
    *,
    stage: int,
    beta: float,
) -> ProposalPath:
    deduction_mix = runtime.options.resolved_family_deduction_mix
    distribution = runtime.candidate_probabilities(
        request,
        batch,
        groups=tuple(family.state_indices for family in choices),
        beta=beta,
        deduction_mix=deduction_mix,
    )
    probabilities = distribution.probabilities
    if seed.target_state_index is None:
        selected = int(categorical_sample(probabilities, 1, generator=runtime.generator)[0].item())
    else:
        target = runtime.support.states[seed.target_state_index]
        target_family = runtime.family_by_hypothesis[target.hypothesis_index]
        try:
            selected = request.candidates.index(family_candidate(target_family))
        except ValueError as error:  # pragma: no cover - support invariant
            raise RuntimeError("target family is absent from finite family support") from error
    probability = float(probabilities[selected].item())
    if not math.isfinite(probability) or probability <= 0.0:
        raise RuntimeError("smoothed family proposal produced nonpositive mass")
    family = choices[selected]
    runtime.record_score_wave(
        stage=stage,
        beta=beta,
        wave="family",
        request=request,
        batch=batch,
        distribution=distribution,
        selected=selected,
        slot=seed.slot,
        ancestor_state_index=seed.ancestor_state_index,
        forced=seed.target_state_index is not None,
        cloned=seed.cloned,
        deduction_mix=deduction_mix,
    )
    runtime.event(
        "importance.proposal.family_selected",
        message="finite structural hypothesis selected",
        level="trace",
        stage=stage,
        slot=seed.slot,
        ancestor_state_index=seed.ancestor_state_index,
        family=family.hypothesis.kind.value,
        probability=probability,
        qwen_probability=float(distribution.q_llm[selected].item()),
        deduction_probability=float(distribution.q_deduction[selected].item()),
        uniform_floor=runtime.options.proposal_epsilon / len(choices),
        deduction_mix=deduction_mix,
        forced=seed.target_state_index is not None,
        cloned=seed.cloned,
    )
    return ProposalPath(
        slot=seed.slot,
        ancestor_state_index=seed.ancestor_state_index,
        family=family,
        compatible_state_indices=family.state_indices,
        cloned=seed.cloned,
        target_state_index=seed.target_state_index,
        log_q_construct=math.log(probability),
    )

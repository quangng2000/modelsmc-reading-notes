"""Exact finite guided proposal and clone-mixture probability accounting."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
    llm_energy,
)
from modelsmc_pbe.smc import categorical_sample

from .deduction_guide import FiniteDeductionGuide
from .prompts import (
    family_candidate,
    family_prompt_prefix,
    hole_prompt_prefix,
    proposal_hole,
)
from .proposal_distribution import CandidateDistribution, candidate_distribution
from .records import (
    FamilySupport,
    HoleFilling,
    ImportanceProposalBudgetExceeded,
    ImportanceSMCOptions,
    ImportanceSupport,
    ProposedState,
)
from .score_ledger import (
    LLMScoreCandidateLedger,
    LLMScoreSelectionLedger,
    LLMScoreWaveLedger,
    prompt_prefix_sha256,
)

type EventEmitter = Callable[..., None]


@dataclass(slots=True)
class _ProposalPath:
    slot: int
    ancestor_state_index: int
    family: FamilySupport
    compatible_state_indices: tuple[int, ...]
    cloned: bool
    target_state_index: int | None
    previous_fillings: list[HoleFilling] = field(default_factory=list)
    log_q_construct: float = 0.0


@dataclass(frozen=True, slots=True)
class _PathSeed:
    slot: int
    ancestor_state_index: int
    cloned: bool
    target_state_index: int | None


@dataclass(frozen=True, slots=True)
class _CandidateChoice:
    key: str
    filling: HoleFilling


@dataclass(frozen=True, slots=True)
class _FamilyChoice:
    key: str
    family: FamilySupport


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _request_batch_sha256(requests: list[CandidateScoreRequest]) -> str:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(request.prompt_prefix.encode("utf-8"))
        digest.update(b"\0")
        for candidate in request.candidates:
            digest.update(candidate.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _deduplicate_requests(
    requests: list[CandidateScoreRequest],
) -> tuple[list[CandidateScoreRequest], tuple[int, ...]]:
    """Collapse identical model-scoring work while retaining every path result."""

    unique: list[CandidateScoreRequest] = []
    positions: dict[tuple[object, ...], int] = {}
    inverse: list[int] = []
    for request in requests:
        key = (
            request.prompt_prefix,
            request.candidates,
            request.hole,
            request.integer_constants,
            request.max_depth,
            request.max_nodes,
            request.candidate_kind,
        )
        position = positions.get(key)
        if position is None:
            position = len(unique)
            positions[key] = position
            unique.append(request)
        inverse.append(position)
    return unique, tuple(inverse)


class FiniteGuidedProposalKernel:
    """Sample canonical construction traces and evaluate their complete ``q``.

    The non-clone proposal is a normalized defensive mixture of Qwen scores,
    exact subtree marginals from sound deduction, and a uniform support floor.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: ImportanceSupport,
        scorer: CandidateScorer,
        generator: torch.Generator,
        emit: EventEmitter | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("importance proposal sampling requires a CPU generator")
        if config.smc.alpha >= 1.0:
            raise ValueError(
                "importance-smc requires alpha < 1 so every target state remains reachable"
            )
        self._config = config
        self._options = options
        self._support = support
        self._scorer = scorer
        self._generator = generator
        self._emit = emit
        self._request_index = 0
        self._scored_candidates = 0
        self._score_ledger: dict[
            tuple[object, ...],
            tuple[LLMScoreWaveLedger, list[LLMScoreSelectionLedger]],
        ] = {}
        self._deduction_guide = FiniteDeductionGuide.build(
            support,
            smc=config.smc,
            strength=options.deduction_strength,
            beta_max=options.beta_max,
        )
        self._deduction_cache: dict[
            tuple[float, tuple[tuple[int, ...], ...]], torch.Tensor
        ] = {}
        self._family_by_hypothesis = {
            family.hypothesis_index: family for family in support.families
        }
        guide = self._deduction_guide.distribution(options.beta_max)
        self._final_deduction_guide = guide
        exact_indices = tuple(
            state.state_index for state in support.states if state.score.exact_program
        )
        self._event(
            "importance.proposal.deduction_guide.ready",
            message="exact Occam/deduction subtree guide constructed",
            level="debug",
            deduction_mix=options.deduction_mix,
            family_deduction_mix=options.resolved_family_deduction_mix,
            hole_deduction_mix=options.resolved_hole_deduction_mix,
            deduction_strength=options.deduction_strength,
            minimum_violations=float(self._deduction_guide.violations.min().item()),
            maximum_violations=float(self._deduction_guide.violations.max().item()),
            exact_program_mass=(
                0.0
                if not exact_indices
                else float(guide[list(exact_indices)].sum().item())
            ),
            family_masses={
                family.hypothesis.kind.value: float(
                    guide[list(family.state_indices)].sum().item()
                )
                for family in support.families
            },
        )

    @property
    def source(self) -> str:
        return self._scorer.name

    @property
    def scored_candidates(self) -> int:
        """Number of unique teacher-forced candidate prompts reserved so far."""

        return self._scored_candidates

    @property
    def final_deduction_guide(self) -> torch.Tensor:
        """Return the normalized final-stage guide in fixed support order."""

        return self._final_deduction_guide.clone()

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        """Freeze all unique scored requests and their sampled uses in call order."""

        return tuple(
            replace(record, selections=tuple(selections))
            for record, selections in self._score_ledger.values()
        )

    async def sample_many(
        self,
        ancestor_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Draw one child per ancestor, including exact clone-mixture mass."""

        seeds = tuple(
            self._sampling_seed(slot, ancestor_state_index)
            for slot, ancestor_state_index in enumerate(ancestor_state_indices)
        )
        paths = await self._select_families(seeds, stage=stage, beta=beta)
        return await self._complete_paths(paths, stage=stage, beta=beta)

    async def evaluate_many(
        self,
        ancestor_state_index: int,
        target_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Evaluate the exact mixture mass of specified target programs."""

        seeds = tuple(
            _PathSeed(
                slot=slot,
                ancestor_state_index=ancestor_state_index,
                cloned=False,
                target_state_index=target_state_index,
            )
            for slot, target_state_index in enumerate(target_state_indices)
        )
        paths = await self._select_families(seeds, stage=stage, beta=beta)
        return await self._complete_paths(paths, stage=stage, beta=beta)

    async def _select_families(
        self,
        seeds: tuple[_PathSeed, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[_ProposalPath, ...]:
        if len(self._support.families) == 1:
            family = self._support.families[0]
            paths: list[_ProposalPath] = []
            for seed in seeds:
                if seed.target_state_index is not None:
                    target = self._support.states[seed.target_state_index]
                    if target.hypothesis_index != family.hypothesis_index:
                        raise RuntimeError("target state is outside the sole viable family")
                paths.append(
                    _ProposalPath(
                        slot=seed.slot,
                        ancestor_state_index=seed.ancestor_state_index,
                        family=family,
                        compatible_state_indices=family.state_indices,
                        cloned=seed.cloned,
                        target_state_index=seed.target_state_index,
                        log_q_construct=0.0,
                    )
                )
            self._event(
                "importance.proposal.family_scoring.skipped",
                message="one viable skeleton family has categorical probability one",
                level="debug",
                stage=stage,
                paths=len(paths),
                family=family.hypothesis.kind.value,
            )
            return tuple(paths)
        requests = [self._family_request(seed, stage=stage, beta=beta) for seed in seeds]
        unique_requests, inverse = _deduplicate_requests(requests)
        request_hash = _request_batch_sha256(unique_requests)
        candidate_total = sum(len(request.candidates) for request in unique_requests)
        self._reserve_candidate_budget(
            candidate_total,
            stage=stage,
            wave="family",
        )
        self._event(
            "importance.proposal.family_scoring.started",
            message="finite skeleton-family scoring wave started",
            level="debug",
            stage=stage,
            requests=len(requests),
            unique_requests=len(unique_requests),
            candidates=candidate_total,
            request_batch_sha256=request_hash,
            source=self.source,
        )
        started = time.perf_counter()
        try:
            unique_batches = await self._scorer.score_many(unique_requests)
        except Exception:
            self._event(
                "importance.proposal.family_scoring.failed",
                message="finite skeleton-family scoring wave failed",
                level="error",
                stage=stage,
                requests=len(requests),
                unique_requests=len(unique_requests),
                candidates=candidate_total,
                elapsed_seconds=time.perf_counter() - started,
                request_batch_sha256=request_hash,
                source=self.source,
            )
            raise
        self._event(
            "importance.proposal.family_scoring.completed",
            message="finite skeleton-family scoring wave completed",
            level="debug",
            stage=stage,
            requests=len(requests),
            unique_requests=len(unique_requests),
            candidates=candidate_total,
            elapsed_seconds=time.perf_counter() - started,
            request_batch_sha256=request_hash,
            source=self.source,
        )
        if len(unique_batches) != len(unique_requests):
            raise RuntimeError("candidate scorer did not return one batch per unique request")
        batches = [unique_batches[position] for position in inverse]
        return tuple(
            self._select_family(seed, request, batch, stage=stage, beta=beta)
            for seed, request, batch in zip(seeds, requests, batches, strict=True)
        )

    async def _complete_paths(
        self,
        paths: tuple[_ProposalPath, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        maximum_holes = max(len(path.family.hypothesis.holes) for path in paths)
        for hole_index in range(maximum_holes):
            active = tuple(path for path in paths if hole_index < len(path.family.hypothesis.holes))
            requests = [
                self._request(path, hole_index=hole_index, stage=stage, beta=beta)
                for path in active
            ]
            unique_requests, inverse = _deduplicate_requests(requests)
            request_hash = _request_batch_sha256(unique_requests)
            candidate_total = sum(len(request.candidates) for request in unique_requests)
            self._reserve_candidate_budget(
                candidate_total,
                stage=stage,
                wave=f"hole-{hole_index}",
            )
            self._event(
                "importance.proposal.scoring.started",
                message="finite-candidate scoring wave started",
                level="debug",
                stage=stage,
                hole_index=hole_index,
                requests=len(requests),
                unique_requests=len(unique_requests),
                candidates=candidate_total,
                request_batch_sha256=request_hash,
                source=self.source,
            )
            started = time.perf_counter()
            try:
                unique_batches = await self._scorer.score_many(unique_requests)
            except Exception:
                self._event(
                    "importance.proposal.scoring.failed",
                    message="finite-candidate scoring wave failed",
                    level="error",
                    stage=stage,
                    hole_index=hole_index,
                    requests=len(requests),
                    unique_requests=len(unique_requests),
                    candidates=candidate_total,
                    elapsed_seconds=time.perf_counter() - started,
                    request_batch_sha256=request_hash,
                    source=self.source,
                )
                raise
            self._event(
                "importance.proposal.scoring.completed",
                message="finite-candidate scoring wave completed",
                level="debug",
                stage=stage,
                hole_index=hole_index,
                requests=len(requests),
                unique_requests=len(unique_requests),
                candidates=candidate_total,
                elapsed_seconds=time.perf_counter() - started,
                request_batch_sha256=request_hash,
                source=self.source,
            )
            if len(unique_batches) != len(unique_requests):
                raise RuntimeError("candidate scorer did not return one batch per unique request")
            batches = [unique_batches[position] for position in inverse]
            for path, request, batch in zip(active, requests, batches, strict=True):
                self._advance(
                    path,
                    hole_index=hole_index,
                    stage=stage,
                    beta=beta,
                    request=request,
                    batch=batch,
                )
        return tuple(self._finish(path) for path in paths)

    def _sampling_seed(self, slot: int, ancestor_state_index: int) -> _PathSeed:
        alpha = self._config.smc.alpha
        cloned = (
            alpha > 0.0
            and float(torch.rand((), dtype=torch.float64, generator=self._generator).item()) < alpha
        )
        return _PathSeed(
            slot=slot,
            ancestor_state_index=ancestor_state_index,
            cloned=cloned,
            target_state_index=ancestor_state_index if cloned else None,
        )

    def _family_choices(self) -> tuple[_FamilyChoice, ...]:
        return tuple(
            sorted(
                (
                    _FamilyChoice(key=family_candidate(family), family=family)
                    for family in self._support.families
                ),
                key=lambda choice: choice.key,
            )
        )

    def _family_request(
        self,
        seed: _PathSeed,
        *,
        stage: int,
        beta: float,
    ) -> CandidateScoreRequest:
        choices = self._family_choices()
        request = CandidateScoreRequest(
            prompt_prefix=family_prompt_prefix(
                config=self._config,
                families=tuple(choice.family for choice in choices),
                ancestor=self._support.states[seed.ancestor_state_index],
                stage=stage,
                beta=beta,
            ),
            candidates=tuple(choice.key for choice in choices),
            hole=None,
            integer_constants=tuple(self._config.spec.integer_constants),
            request_index=self._request_index,
            max_depth=self._config.smc.max_depth,
            max_nodes=self._config.smc.max_nodes,
            candidate_kind=CandidateKind.SKELETON,
        )
        self._request_index += 1
        return request

    def _select_family(
        self,
        seed: _PathSeed,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        *,
        stage: int,
        beta: float,
    ) -> _ProposalPath:
        choices = self._family_choices()
        deduction_mix = self._options.resolved_family_deduction_mix
        distribution = self._candidate_probabilities(
            request,
            batch,
            groups=tuple(choice.family.state_indices for choice in choices),
            beta=beta,
            deduction_mix=deduction_mix,
        )
        probabilities = distribution.probabilities
        if seed.target_state_index is None:
            selected = int(
                categorical_sample(
                    probabilities,
                    1,
                    generator=self._generator,
                )[0].item()
            )
        else:
            target = self._support.states[seed.target_state_index]
            target_family = self._family_by_hypothesis[target.hypothesis_index]
            target_key = family_candidate(target_family)
            try:
                selected = request.candidates.index(target_key)
            except ValueError as error:  # pragma: no cover - support invariant
                raise RuntimeError("target family is absent from finite family support") from error
        probability = float(probabilities[selected].item())
        if not math.isfinite(probability) or probability <= 0.0:
            raise RuntimeError("smoothed family proposal produced nonpositive mass")
        family = choices[selected].family
        self._record_score_wave(
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
        self._event(
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
            uniform_floor=self._options.proposal_epsilon / len(choices),
            deduction_mix=deduction_mix,
            forced=seed.target_state_index is not None,
            cloned=seed.cloned,
        )
        return _ProposalPath(
            slot=seed.slot,
            ancestor_state_index=seed.ancestor_state_index,
            family=family,
            compatible_state_indices=family.state_indices,
            cloned=seed.cloned,
            target_state_index=seed.target_state_index,
            log_q_construct=math.log(probability),
        )

    def _choices(self, path: _ProposalPath, hole_index: int) -> tuple[_CandidateChoice, ...]:
        hole_name = path.family.hypothesis.holes[hole_index].name
        unique: dict[str, HoleFilling] = {}
        for state_index in path.compatible_state_indices:
            filling = self._support.states[state_index].filling(hole_name)
            unique.setdefault(filling.key, filling)
        return tuple(_CandidateChoice(key=key, filling=unique[key]) for key in sorted(unique))

    def _request(
        self,
        path: _ProposalPath,
        *,
        hole_index: int,
        stage: int,
        beta: float,
    ) -> CandidateScoreRequest:
        choices = self._choices(path, hole_index)
        hole = path.family.hypothesis.holes[hole_index]
        ancestor = self._support.states[path.ancestor_state_index]
        signature = self._config.spec.signature
        if signature is None:  # pragma: no cover
            raise ValueError("PBE specification has no signature")
        request = CandidateScoreRequest(
            prompt_prefix=hole_prompt_prefix(
                config=self._config,
                report=path.family.deduction,
                hole=hole,
                ancestor=ancestor,
                previous_fillings=path.previous_fillings,
                stage=stage,
                beta=beta,
                candidate_count=len(choices),
            ),
            candidates=tuple(choice.key for choice in choices),
            hole=proposal_hole(hole, input_type=signature.input_type),
            integer_constants=tuple(self._config.spec.integer_constants),
            request_index=self._request_index,
            max_depth=self._config.smc.max_depth,
            max_nodes=self._config.smc.max_nodes,
        )
        self._request_index += 1
        return request

    def _advance(
        self,
        path: _ProposalPath,
        *,
        hole_index: int,
        stage: int,
        beta: float,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
    ) -> None:
        choices = self._choices(path, hole_index)
        hole_name = path.family.hypothesis.holes[hole_index].name
        groups = tuple(
            tuple(
                state_index
                for state_index in path.compatible_state_indices
                if self._support.states[state_index].filling(hole_name).key == choice.key
            )
            for choice in choices
        )
        deduction_mix = self._options.resolved_hole_deduction_mix
        distribution = self._candidate_probabilities(
            request,
            batch,
            groups=groups,
            beta=beta,
            deduction_mix=deduction_mix,
        )
        probabilities = distribution.probabilities
        if path.target_state_index is None:
            selected = int(
                categorical_sample(
                    probabilities,
                    1,
                    generator=self._generator,
                )[0].item()
            )
        else:
            target = self._support.states[path.target_state_index]
            target_key = target.filling(path.family.hypothesis.holes[hole_index].name).key
            try:
                selected = request.candidates.index(target_key)
            except ValueError as error:  # pragma: no cover - support-trie invariant
                raise RuntimeError("target state is absent from its construction trie") from error
        probability = float(probabilities[selected].item())
        if not math.isfinite(probability) or probability <= 0.0:
            raise RuntimeError("smoothed finite proposal produced nonpositive mass")
        chosen = choices[selected].filling
        self._record_score_wave(
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
        self._event(
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
            uniform_floor=self._options.proposal_epsilon / len(choices),
            deduction_mix=deduction_mix,
            deduction_examples=len(
                path.family.deduction.examples_for(chosen.hole_name)
            ),
            forced=path.target_state_index is not None,
            cloned=path.cloned,
        )
        path.log_q_construct += math.log(probability)
        path.previous_fillings.append(chosen)
        path.compatible_state_indices = tuple(
            state_index
            for state_index in path.compatible_state_indices
            if self._support.states[state_index].filling(chosen.hole_name).key == chosen.key
        )
        if not path.compatible_state_indices:  # pragma: no cover - trie invariant
            raise RuntimeError("proposal choice has no valid complete program")

    def _candidate_probabilities(
        self,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        *,
        groups: tuple[tuple[int, ...], ...],
        beta: float,
        deduction_mix: float,
    ) -> CandidateDistribution:
        if tuple(score.candidate for score in batch.scores) != request.candidates:
            raise RuntimeError("candidate scorer reordered or changed the finite catalog")
        q_deduction = self._deduction_probabilities(groups, beta=beta)
        return candidate_distribution(
            sequence_logprobs=tuple(
                llm_energy(score, self._options.llm_energy_normalization)
                for score in batch.scores
            ),
            q_deduction=q_deduction,
            temperature=self._options.proposal_temperature,
            epsilon=self._options.proposal_epsilon,
            deduction_mix=deduction_mix,
        )

    def _record_score_wave(
        self,
        *,
        stage: int,
        beta: float,
        wave: str,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        distribution: CandidateDistribution,
        selected: int,
        slot: int,
        ancestor_state_index: int,
        forced: bool,
        cloned: bool,
        deduction_mix: float,
    ) -> None:
        """Deduplicate score evidence while retaining every categorical use."""

        qwen = tuple(float(value) for value in distribution.q_llm.tolist())
        deduction = tuple(float(value) for value in distribution.q_deduction.tolist())
        proposal = tuple(float(value) for value in distribution.probabilities.tolist())
        key = (
            stage,
            wave,
            request.prompt_prefix,
            request.candidates,
            batch.source,
            batch.model,
            batch.model_revision,
            batch.tokenizer_revision,
            batch.semantics,
            self._options.llm_energy_normalization,
            self._options.proposal_temperature,
            self._options.proposal_epsilon,
            deduction_mix,
            deduction,
            proposal,
        )
        entry = self._score_ledger.get(key)
        if entry is None:
            candidates = tuple(
                LLMScoreCandidateLedger(
                    canonical_candidate=score.candidate,
                    token_ids=score.token_ids,
                    token_logprobs=score.token_logprobs,
                    scored_token_count=len(score.token_logprobs),
                    total_sequence_logprob=score.sequence_logprob,
                    normalized_energy=llm_energy(
                        score, self._options.llm_energy_normalization
                    ),
                    qwen_probability=qwen[index],
                    deduction_probability=deduction[index],
                    proposal_probability=proposal[index],
                )
                for index, score in enumerate(batch.scores)
            )
            record = LLMScoreWaveLedger(
                stage=stage,
                beta=beta,
                wave=wave,
                request_index=request.request_index,
                prompt_prefix=request.prompt_prefix,
                prompt_prefix_sha256=prompt_prefix_sha256(request.prompt_prefix),
                candidate_kind=request.candidate_kind.value,
                source=batch.source,
                model=batch.model,
                model_revision=batch.model_revision,
                tokenizer_revision=batch.tokenizer_revision,
                score_semantics=batch.semantics.value,
                score_origin=(
                    "unspecified"
                    if batch.provenance is None
                    else batch.provenance.origin.value
                ),
                cache_key_sha256=(
                    None
                    if batch.provenance is None
                    else batch.provenance.cache_key_sha256
                ),
                cache_hit=(
                    None if batch.provenance is None else batch.provenance.cache_hit
                ),
                energy_normalization=self._options.llm_energy_normalization,
                temperature=self._options.proposal_temperature,
                proposal_epsilon=self._options.proposal_epsilon,
                deduction_mix=deduction_mix,
                candidates=candidates,
                selections=(),
            )
            entry = (record, [])
            self._score_ledger[key] = entry
        entry[1].append(
            LLMScoreSelectionLedger(
                slot=slot,
                ancestor_state_index=ancestor_state_index,
                selected_index=selected,
                selected_probability=proposal[selected],
                selected_qwen_probability=qwen[selected],
                selected_deduction_probability=deduction[selected],
                forced=forced,
                cloned=cloned,
            )
        )

    def _deduction_probabilities(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        beta: float,
    ) -> torch.Tensor:
        cache_key = (beta, groups)
        cached = self._deduction_cache.get(cache_key)
        if cached is not None:
            return cached
        probabilities = self._deduction_guide.subtree_probabilities(groups, beta=beta)
        self._deduction_cache[cache_key] = probabilities
        return probabilities

    def _finish(self, path: _ProposalPath) -> ProposedState:
        if len(path.compatible_state_indices) != 1:
            raise RuntimeError(
                "construction trace is not one-to-one with a canonical complete program"
            )
        state_index = path.compatible_state_indices[0]
        if path.target_state_index is not None and state_index != path.target_state_index:
            raise RuntimeError("probability evaluation did not recover its target state")
        alpha = self._config.smc.alpha
        if state_index == path.ancestor_state_index and alpha > 0.0:
            log_q_mixture = _logaddexp(
                math.log(alpha),
                math.log1p(-alpha) + path.log_q_construct,
            )
        elif alpha > 0.0:
            log_q_mixture = math.log1p(-alpha) + path.log_q_construct
        else:
            log_q_mixture = path.log_q_construct
        return ProposedState(
            state_index=state_index,
            ancestor_state_index=path.ancestor_state_index,
            log_q_construct=path.log_q_construct,
            log_q_mixture=log_q_mixture,
            cloned=path.cloned,
            family=path.family.hypothesis.kind.value,
        )

    def _event(
        self,
        name: str,
        *,
        message: str,
        level: str,
        **data: Any,
    ) -> None:
        if self._emit is not None:
            self._emit(name, message=message, level=level, **data)

    def _reserve_candidate_budget(
        self,
        candidates: int,
        *,
        stage: int,
        wave: str,
    ) -> None:
        attempted = self._scored_candidates + candidates
        if attempted > self._options.max_scored_candidates:
            self._event(
                "importance.proposal.budget_exceeded",
                message="candidate-scoring wave rejected before scorer I/O",
                level="error",
                stage=stage,
                wave=wave,
                requested_candidates=candidates,
                scored_candidates=self._scored_candidates,
                max_scored_candidates=self._options.max_scored_candidates,
            )
            raise ImportanceProposalBudgetExceeded(
                "candidate scoring would exceed --max-scored-candidates="
                f"{self._options.max_scored_candidates} "
                f"(used {self._scored_candidates}, next wave {candidates})"
            )
        self._scored_candidates = attempted
        self._event(
            "importance.proposal.budget_reserved",
            message="candidate-scoring work reserved within the run budget",
            level="debug",
            stage=stage,
            wave=wave,
            requested_candidates=candidates,
            scored_candidates=self._scored_candidates,
            max_scored_candidates=self._options.max_scored_candidates,
        )

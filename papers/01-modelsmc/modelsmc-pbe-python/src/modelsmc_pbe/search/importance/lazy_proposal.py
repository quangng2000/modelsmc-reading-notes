"""Known-density guided proposals over factorized typed hole catalogs."""

from __future__ import annotations

import hashlib
import json
import math
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

from .factorized import (
    choice_guide_probabilities,
    family_guide_probabilities,
    valid_choice_indices,
)
from .lazy_prompts import lazy_family_prompt_prefix, lazy_hole_prompt_prefix
from .lazy_records import (
    ConstructionTrace,
    FactorizedFamily,
    FactorizedImportanceSupport,
    LazyImportanceState,
    LazyProposedTrace,
)
from .prompts import proposal_hole
from .proposal_distribution import CandidateDistribution, candidate_distribution
from .records import HoleFilling, ImportanceProposalBudgetExceeded, ImportanceSMCOptions
from .score_ledger import (
    LLMScoreCandidateLedger,
    LLMScoreSelectionLedger,
    LLMScoreWaveLedger,
    prompt_prefix_sha256,
)

type EventEmitter = Callable[..., None]


@dataclass(frozen=True, slots=True)
class _Seed:
    slot: int
    ancestor: LazyImportanceState
    cloned: bool
    target_trace: ConstructionTrace | None


@dataclass(slots=True)
class _Path:
    slot: int
    ancestor: LazyImportanceState
    family: FactorizedFamily
    cloned: bool
    target_trace: ConstructionTrace | None
    selected_indices: list[int] = field(default_factory=list)
    previous_fillings: list[HoleFilling] = field(default_factory=list)
    prefix_cost: int = 0
    log_q_construct: float = 0.0


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _request_key(request: CandidateScoreRequest) -> tuple[object, ...]:
    return (
        request.prompt_prefix,
        request.candidates,
        request.hole,
        request.integer_constants,
        request.max_depth,
        request.max_nodes,
        request.candidate_kind,
    )


def _deduplicate_requests(
    requests: list[CandidateScoreRequest],
) -> tuple[list[CandidateScoreRequest], tuple[int, ...]]:
    unique: list[CandidateScoreRequest] = []
    positions: dict[tuple[object, ...], int] = {}
    inverse: list[int] = []
    for request in requests:
        key = _request_key(request)
        position = positions.get(key)
        if position is None:
            position = len(unique)
            positions[key] = position
            unique.append(request)
        inverse.append(position)
    return unique, tuple(inverse)


class LazyGuidedProposalKernel:
    """Sample factorized traces without consulting a complete-state table."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: FactorizedImportanceSupport,
        scorer: CandidateScorer,
        generator: torch.Generator,
        emit: EventEmitter | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("lazy importance proposal requires a CPU generator")
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

    @property
    def source(self) -> str:
        return self._scorer.name

    @property
    def scored_candidates(self) -> int:
        return self._scored_candidates

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        """Freeze unique scored requests and all of their categorical uses."""

        return tuple(
            replace(record, selections=tuple(selections))
            for record, selections in self._score_ledger.values()
        )

    def final_family_guide(self) -> torch.Tensor:
        """Return exact factorized guide mass by family at beta-max."""

        return family_guide_probabilities(
            self._support,
            cost_scale=float(self._config.smc.cost_scale),
            violation_scale=self._options.deduction_strength,
        )

    async def sample_many(
        self,
        ancestors: tuple[LazyImportanceState, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[LazyProposedTrace, ...]:
        seeds = tuple(self._seed(slot, ancestor) for slot, ancestor in enumerate(ancestors))
        paths = await self._select_families(seeds, stage=stage, beta=beta)
        for hole_index in range(max(len(path.family.catalogs) for path in paths)):
            active = [path for path in paths if hole_index < len(path.family.catalogs)]
            requests = [
                self._hole_request(path, hole_index, stage=stage, beta=beta)
                for path in active
            ]
            batches = await self._score_wave(
                requests,
                stage=stage,
                wave=f"hole-{hole_index}",
            )
            for path, request, batch in zip(active, requests, batches, strict=True):
                self._advance_hole(
                    path,
                    hole_index,
                    request,
                    batch,
                    stage=stage,
                    beta=beta,
                )
        return tuple(self._finish(path) for path in paths)

    def _seed(self, slot: int, ancestor: LazyImportanceState) -> _Seed:
        alpha = self._config.smc.alpha
        cloned = alpha > 0.0 and float(
            torch.rand((), dtype=torch.float64, generator=self._generator).item()
        ) < alpha
        return _Seed(
            slot=slot,
            ancestor=ancestor,
            cloned=cloned,
            target_trace=ancestor.trace if cloned else None,
        )

    async def _select_families(
        self,
        seeds: tuple[_Seed, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[_Path, ...]:
        choices = tuple(sorted(self._support.families, key=lambda item: item.hypothesis.kind.value))
        if len(choices) == 1:
            return tuple(
                _Path(
                    slot=seed.slot,
                    ancestor=seed.ancestor,
                    family=choices[0],
                    cloned=seed.cloned,
                    target_trace=seed.target_trace,
                )
                for seed in seeds
            )
        requests = [self._family_request(seed, choices, stage=stage, beta=beta) for seed in seeds]
        batches = await self._score_wave(requests, stage=stage, wave="family")
        support_guide = family_guide_probabilities(
            self._support,
            cost_scale=float(self._config.smc.cost_scale),
            violation_scale=self._violation_scale(beta),
        )
        guide_by_index = {
            family.hypothesis_index: float(support_guide[index].item())
            for index, family in enumerate(self._support.families)
        }
        q_deduction = torch.tensor(
            [guide_by_index[family.hypothesis_index] for family in choices],
            dtype=torch.float64,
        )
        paths: list[_Path] = []
        for seed, request, batch in zip(seeds, requests, batches, strict=True):
            deduction_mix = self._options.resolved_family_deduction_mix
            distribution = self._distribution(
                request,
                batch,
                q_deduction,
                deduction_mix=deduction_mix,
            )
            if seed.target_trace is None:
                position = int(
                    categorical_sample(
                        distribution.probabilities,
                        1,
                        generator=self._generator,
                    )[0].item()
                )
            else:
                position = next(
                    index
                    for index, family in enumerate(choices)
                    if family.hypothesis_index == seed.target_trace.hypothesis_index
                )
            probability = float(distribution.probabilities[position].item())
            family = choices[position]
            self._record_score_wave(
                stage=stage,
                beta=beta,
                wave="family",
                request=request,
                batch=batch,
                distribution=distribution,
                selected=position,
                slot=seed.slot,
                forced=seed.target_trace is not None,
                cloned=seed.cloned,
                deduction_mix=deduction_mix,
            )
            paths.append(
                _Path(
                    slot=seed.slot,
                    ancestor=seed.ancestor,
                    family=family,
                    cloned=seed.cloned,
                    target_trace=seed.target_trace,
                    log_q_construct=math.log(probability),
                )
            )
            self._event(
                "importance.lazy.proposal.family_selected",
                message="sampled one factorized skeleton family",
                level="trace",
                stage=stage,
                slot=seed.slot,
                family=family.hypothesis.kind.value,
                probability=probability,
                forced=seed.target_trace is not None,
            )
        return tuple(paths)

    def _family_request(
        self,
        seed: _Seed,
        choices: tuple[FactorizedFamily, ...],
        *,
        stage: int,
        beta: float,
    ) -> CandidateScoreRequest:
        request = CandidateScoreRequest(
            prompt_prefix=lazy_family_prompt_prefix(
                config=self._config,
                families=choices,
                ancestor=seed.ancestor,
                stage=stage,
                beta=beta,
            ),
            candidates=tuple(
                json.dumps(family.hypothesis.kind.value, ensure_ascii=False)
                for family in choices
            ),
            hole=None,
            integer_constants=tuple(self._config.spec.integer_constants),
            request_index=self._request_index,
            max_depth=self._config.smc.max_depth,
            max_nodes=self._config.smc.max_nodes,
            candidate_kind=CandidateKind.SKELETON,
        )
        self._request_index += 1
        return request

    def _hole_request(
        self,
        path: _Path,
        hole_index: int,
        *,
        stage: int,
        beta: float,
    ) -> CandidateScoreRequest:
        family = path.family
        catalog = family.catalogs[hole_index]
        choices = valid_choice_indices(
            family,
            hole_index=hole_index,
            prefix_cost=path.prefix_cost,
        )
        signature = self._config.spec.signature
        if signature is None:  # pragma: no cover
            raise ValueError("PBE specification has no signature")
        request = CandidateScoreRequest(
            prompt_prefix=lazy_hole_prompt_prefix(
                config=self._config,
                family=family,
                hole_index=hole_index,
                ancestor=path.ancestor,
                previous_fillings=path.previous_fillings,
                stage=stage,
                beta=beta,
                candidate_count=len(choices),
            ),
            candidates=tuple(catalog.fillings[index].key for index in choices),
            hole=proposal_hole(catalog.hole, input_type=signature.input_type),
            integer_constants=tuple(self._config.spec.integer_constants),
            request_index=self._request_index,
            max_depth=self._config.smc.max_depth,
            max_nodes=self._config.smc.max_nodes,
        )
        self._request_index += 1
        return request

    def _advance_hole(
        self,
        path: _Path,
        hole_index: int,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        *,
        stage: int,
        beta: float,
    ) -> None:
        family = path.family
        catalog = family.catalogs[hole_index]
        choices = valid_choice_indices(
            family,
            hole_index=hole_index,
            prefix_cost=path.prefix_cost,
        )
        q_deduction = choice_guide_probabilities(
            family,
            hole_index=hole_index,
            prefix_cost=path.prefix_cost,
            choice_indices=choices,
            cost_scale=float(self._config.smc.cost_scale),
            violation_scale=self._violation_scale(beta),
        )
        deduction_mix = self._options.resolved_hole_deduction_mix
        distribution = self._distribution(
            request,
            batch,
            q_deduction,
            deduction_mix=deduction_mix,
        )
        if path.target_trace is None:
            position = int(
                categorical_sample(
                    distribution.probabilities,
                    1,
                    generator=self._generator,
                )[0].item()
            )
        else:
            target_choice = path.target_trace.filling_indices[hole_index]
            position = choices.index(target_choice)
        choice = choices[position]
        probability = float(distribution.probabilities[position].item())
        filling = catalog.fillings[choice]
        self._record_score_wave(
            stage=stage,
            beta=beta,
            wave=f"hole-{hole_index}",
            request=request,
            batch=batch,
            distribution=distribution,
            selected=position,
            slot=path.slot,
            forced=path.target_trace is not None,
            cloned=path.cloned,
            deduction_mix=deduction_mix,
        )
        path.selected_indices.append(choice)
        path.previous_fillings.append(filling)
        path.prefix_cost += catalog.costs[choice]
        path.log_q_construct += math.log(probability)
        self._event(
            "importance.lazy.proposal.hole_selected",
            message="sampled one typed hole without assembling a complete program",
            level="trace",
            stage=stage,
            slot=path.slot,
            hole=catalog.hole.name,
            probability=probability,
            candidate_sha256=hashlib.sha256(filling.key.encode()).hexdigest(),
            forced=path.target_trace is not None,
        )

    def _finish(self, path: _Path) -> LazyProposedTrace:
        trace = ConstructionTrace(
            hypothesis_index=path.family.hypothesis_index,
            filling_indices=tuple(path.selected_indices),
        )
        if path.target_trace is not None and trace != path.target_trace:
            raise RuntimeError("forced clone trace was not recovered")
        alpha = self._config.smc.alpha
        if alpha > 0.0 and trace == path.ancestor.trace:
            log_q_mixture = _logaddexp(
                math.log(alpha),
                math.log1p(-alpha) + path.log_q_construct,
            )
        elif alpha > 0.0:
            log_q_mixture = math.log1p(-alpha) + path.log_q_construct
        else:
            log_q_mixture = path.log_q_construct
        return LazyProposedTrace(
            trace=trace,
            ancestor_trace=path.ancestor.trace,
            log_q_construct=path.log_q_construct,
            log_q_mixture=log_q_mixture,
            cloned=path.cloned,
            family=path.family.hypothesis.kind.value,
        )

    def _distribution(
        self,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        q_deduction: torch.Tensor,
        *,
        deduction_mix: float,
    ) -> CandidateDistribution:
        if tuple(score.candidate for score in batch.scores) != request.candidates:
            raise RuntimeError("candidate scorer reordered or changed the finite catalog")
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
                ancestor_state_index=None,
                selected_index=selected,
                selected_probability=proposal[selected],
                selected_qwen_probability=qwen[selected],
                selected_deduction_probability=deduction[selected],
                forced=forced,
                cloned=cloned,
            )
        )

    def _violation_scale(self, beta: float) -> float:
        return self._options.deduction_strength * beta / self._options.beta_max

    async def _score_wave(
        self,
        requests: list[CandidateScoreRequest],
        *,
        stage: int,
        wave: str,
    ) -> list[CandidateScoreBatch]:
        unique, inverse = _deduplicate_requests(requests)
        candidates = sum(len(request.candidates) for request in unique)
        attempted = self._scored_candidates + candidates
        if attempted > self._options.max_scored_candidates:
            raise ImportanceProposalBudgetExceeded(
                f"candidate scoring would exceed --max-scored-candidates="
                f"{self._options.max_scored_candidates}"
            )
        self._scored_candidates = attempted
        self._event(
            "importance.lazy.proposal.scoring.started",
            message="factorized finite-candidate scoring wave started",
            level="debug",
            stage=stage,
            wave=wave,
            requests=len(requests),
            unique_requests=len(unique),
            candidates=candidates,
        )
        batches = await self._scorer.score_many(unique)
        if len(batches) != len(unique):
            raise RuntimeError("candidate scorer did not return one batch per request")
        return [batches[position] for position in inverse]

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

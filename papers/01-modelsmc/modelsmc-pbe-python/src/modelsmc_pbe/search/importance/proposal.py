"""Exact finite categorical proposal and clone-mixture probability accounting."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.proposals import (
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
)
from modelsmc_pbe.smc import categorical_sample, normalize_log_weights

from .prompts import hole_prompt_prefix, proposal_hole
from .records import (
    FamilySupport,
    HoleFilling,
    ImportanceSMCOptions,
    ImportanceSupport,
    ProposedState,
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
    log_q_llm: float = 0.0


@dataclass(frozen=True, slots=True)
class _CandidateChoice:
    key: str
    filling: HoleFilling


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


class FiniteLLMProposalKernel:
    """Sample canonical construction traces and evaluate their complete ``q``."""

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
        self._family_by_hypothesis = {
            family.hypothesis_index: family for family in support.families
        }

    @property
    def source(self) -> str:
        return self._scorer.name

    async def sample_many(
        self,
        ancestor_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Draw one child per ancestor, including exact clone-mixture mass."""

        paths = tuple(
            self._start_path(slot, ancestor_state_index)
            for slot, ancestor_state_index in enumerate(ancestor_state_indices)
        )
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

        paths = tuple(
            self._evaluation_path(slot, ancestor_state_index, target_state_index)
            for slot, target_state_index in enumerate(target_state_indices)
        )
        return await self._complete_paths(paths, stage=stage, beta=beta)

    async def _complete_paths(
        self,
        paths: tuple[_ProposalPath, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        maximum_holes = max(len(path.family.hypothesis.holes) for path in paths)
        for hole_index in range(maximum_holes):
            active = tuple(
                path for path in paths if hole_index < len(path.family.hypothesis.holes)
            )
            requests = [
                self._request(path, hole_index=hole_index, stage=stage, beta=beta)
                for path in active
            ]
            request_hash = _request_batch_sha256(requests)
            candidate_total = sum(len(request.candidates) for request in requests)
            self._event(
                "importance.proposal.scoring.started",
                message="finite-candidate scoring wave started",
                level="debug",
                stage=stage,
                hole_index=hole_index,
                requests=len(requests),
                candidates=candidate_total,
                request_batch_sha256=request_hash,
                source=self.source,
            )
            started = time.perf_counter()
            try:
                batches = await self._scorer.score_many(requests)
            except Exception:
                self._event(
                    "importance.proposal.scoring.failed",
                    message="finite-candidate scoring wave failed",
                    level="error",
                    stage=stage,
                    hole_index=hole_index,
                    requests=len(requests),
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
                candidates=candidate_total,
                elapsed_seconds=time.perf_counter() - started,
                request_batch_sha256=request_hash,
                source=self.source,
            )
            if len(batches) != len(active):
                raise RuntimeError("candidate scorer did not return one batch per request")
            for path, request, batch in zip(active, requests, batches, strict=True):
                self._advance(path, hole_index=hole_index, request=request, batch=batch)
        return tuple(self._finish(path) for path in paths)

    def _evaluation_path(
        self,
        slot: int,
        ancestor_state_index: int,
        target_state_index: int,
    ) -> _ProposalPath:
        target = self._support.states[target_state_index]
        family = self._family_by_hypothesis[target.hypothesis_index]
        return _ProposalPath(
            slot=slot,
            ancestor_state_index=ancestor_state_index,
            family=family,
            compatible_state_indices=family.state_indices,
            cloned=False,
            target_state_index=target_state_index,
            log_q_llm=-math.log(len(self._support.families)),
        )

    def _start_path(self, slot: int, ancestor_state_index: int) -> _ProposalPath:
        ancestor = self._support.states[ancestor_state_index]
        alpha = self._config.smc.alpha
        cloned = alpha > 0.0 and float(
            torch.rand((), dtype=torch.float64, generator=self._generator).item()
        ) < alpha
        if cloned:
            family = self._family_by_hypothesis[ancestor.hypothesis_index]
            target_state_index: int | None = ancestor_state_index
        else:
            family_index = int(
                torch.randint(
                    len(self._support.families),
                    (),
                    dtype=torch.int64,
                    generator=self._generator,
                ).item()
            )
            family = self._support.families[family_index]
            target_state_index = None
        return _ProposalPath(
            slot=slot,
            ancestor_state_index=ancestor_state_index,
            family=family,
            compatible_state_indices=family.state_indices,
            cloned=cloned,
            target_state_index=target_state_index,
            log_q_llm=-math.log(len(self._support.families)),
        )

    def _choices(self, path: _ProposalPath, hole_index: int) -> tuple[_CandidateChoice, ...]:
        hole_name = path.family.hypothesis.holes[hole_index].name
        unique: dict[str, HoleFilling] = {}
        for state_index in path.compatible_state_indices:
            filling = self._support.states[state_index].filling(hole_name)
            unique.setdefault(filling.key, filling)
        return tuple(
            _CandidateChoice(key=key, filling=unique[key]) for key in sorted(unique)
        )

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
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
    ) -> None:
        if tuple(score.candidate for score in batch.scores) != request.candidates:
            raise RuntimeError("candidate scorer reordered or changed the finite catalog")
        logits = torch.tensor(
            [score.sequence_logprob for score in batch.scores], dtype=torch.float64
        ) / self._options.proposal_temperature
        softmax = normalize_log_weights(logits).weights
        count = int(softmax.numel())
        probabilities = (
            (1.0 - self._options.proposal_epsilon) * softmax
            + self._options.proposal_epsilon / count
        )
        probabilities = probabilities / probabilities.sum()
        choices = self._choices(path, hole_index)
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
        path.log_q_llm += math.log(probability)
        path.previous_fillings.append(chosen)
        path.compatible_state_indices = tuple(
            state_index
            for state_index in path.compatible_state_indices
            if self._support.states[state_index].filling(chosen.hole_name).key == chosen.key
        )
        if not path.compatible_state_indices:  # pragma: no cover - trie invariant
            raise RuntimeError("proposal choice has no valid complete program")

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
                math.log1p(-alpha) + path.log_q_llm,
            )
        elif alpha > 0.0:
            log_q_mixture = math.log1p(-alpha) + path.log_q_llm
        else:
            log_q_mixture = path.log_q_llm
        return ProposedState(
            state_index=state_index,
            ancestor_state_index=path.ancestor_state_index,
            log_q_llm=path.log_q_llm,
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

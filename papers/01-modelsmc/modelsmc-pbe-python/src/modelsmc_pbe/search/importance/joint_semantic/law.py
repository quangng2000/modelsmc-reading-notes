"""Exact defensive proposal law over a factorized semantic trace slate."""

from __future__ import annotations

import math

import torch

from ..factorized import choice_guide_probabilities, trace_log_prior, valid_choice_indices
from ..lazy_records import ConstructionTrace, FactorizedFamily, FactorizedImportanceSupport
from .records import (
    SemanticBranchDistribution,
    SemanticSlateProbability,
    SemanticTracePath,
)
from .slate import SemanticSlateEntry

type PrefixKey = tuple[int, tuple[int, ...]]


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


class SemanticStageLaw:
    """Normalized ``epsilon*pi + (1-epsilon)*q_A`` construction law.

    The semantic component is restricted to the scored slate while the exact
    factorized prior supplies positive mass to every trace in the full support.
    Prefix conditionals are computed from prior suffix partitions and semantic
    prefix log masses, so their product telescopes to the direct leaf mixture.
    """

    def __init__(
        self,
        *,
        support: FactorizedImportanceSupport,
        slate: tuple[SemanticSlateEntry, ...],
        cost_scale: float,
        semantic_scale: float,
        stage_fraction: float,
        epsilon: float,
    ) -> None:
        self._validate_controls(
            cost_scale=cost_scale,
            semantic_scale=semantic_scale,
            stage_fraction=stage_fraction,
            epsilon=epsilon,
        )
        if not slate:
            raise ValueError("semantic proposal requires a nonempty slate")
        if len({entry.trace for entry in slate}) != len(slate):
            raise ValueError("semantic slate traces must be unique")
        self.support = support
        self.slate = slate
        self.cost_scale = cost_scale
        self.semantic_scale = semantic_scale
        self.stage_fraction = stage_fraction
        self.epsilon = epsilon
        self._log_epsilon = math.log(epsilon)
        self._log_semantic_mix = -math.inf if epsilon == 1.0 else math.log1p(-epsilon)
        self._prior_prefix_cache: dict[PrefixKey, float] = {}
        self._validate_slate_priors()
        self._log_q_a = self._normalize_semantic_component()
        self._semantic_prefixes = self._build_semantic_prefixes()

    @staticmethod
    def _validate_controls(
        *,
        cost_scale: float,
        semantic_scale: float,
        stage_fraction: float,
        epsilon: float,
    ) -> None:
        if not math.isfinite(cost_scale) or cost_scale < 0.0:
            raise ValueError("cost_scale must be finite and nonnegative")
        if not math.isfinite(semantic_scale) or semantic_scale < 0.0:
            raise ValueError("semantic_scale must be finite and nonnegative")
        if not math.isfinite(stage_fraction) or not 0.0 <= stage_fraction <= 1.0:
            raise ValueError("stage_fraction must be finite and in [0, 1]")
        if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be finite and in (0, 1]")

    def _validate_slate_priors(self) -> None:
        for entry in self.slate:
            expected = trace_log_prior(
                self.support,
                entry.trace,
                cost_scale=self.cost_scale,
            )
            if not math.isclose(entry.log_prior, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("semantic slate log prior disagrees with factorized support")

    def _normalize_semantic_component(self) -> tuple[float, ...]:
        factor = self.stage_fraction * self.semantic_scale
        maximum_score = max(entry.semantic_score for entry in self.slate)
        logits = tuple(
            entry.log_prior
            + (0.0 if factor == 0.0 else factor * (entry.semantic_score - maximum_score))
            for entry in self.slate
        )
        log_normalizer = _logsumexp(logits)
        if not math.isfinite(log_normalizer):
            raise ValueError("semantic slate has no finite normalized mass")
        return tuple(logit - log_normalizer for logit in logits)

    def _build_semantic_prefixes(self) -> dict[PrefixKey, float]:
        prefixes: dict[PrefixKey, float] = {}
        for entry, log_probability in zip(self.slate, self._log_q_a, strict=True):
            trace = entry.trace
            for length in range(len(trace.filling_indices) + 1):
                key = (trace.hypothesis_index, trace.filling_indices[:length])
                prefixes[key] = _logaddexp(prefixes.get(key, -math.inf), log_probability)
        return prefixes

    @property
    def slate_probabilities(self) -> tuple[SemanticSlateProbability, ...]:
        """Return auditable per-slate normalized semantic and mixture masses."""

        return tuple(
            SemanticSlateProbability(
                entry=entry,
                log_q_a=log_q_a,
                log_q=self.direct_log_probability(entry.trace),
            )
            for entry, log_q_a in zip(self.slate, self._log_q_a, strict=True)
        )

    def q_a_log_probability(self, trace: ConstructionTrace) -> float:
        """Return semantic-component leaf mass, or ``-inf`` outside the slate."""

        return self._semantic_prefixes.get(
            (trace.hypothesis_index, trace.filling_indices),
            -math.inf,
        )

    def _prior_prefix_log_mass(
        self,
        family: FactorizedFamily,
        prefix: tuple[int, ...],
    ) -> float:
        key = (family.hypothesis_index, prefix)
        cached = self._prior_prefix_cache.get(key)
        if cached is not None:
            return cached
        log_mass = -math.log(len(self.support.families))
        prefix_cost = 0
        for hole_index, choice in enumerate(prefix):
            choices = valid_choice_indices(
                family,
                hole_index=hole_index,
                prefix_cost=prefix_cost,
            )
            if choice not in choices:
                raise ValueError("trace prefix is invalid under the complete-program budget")
            probabilities = choice_guide_probabilities(
                family,
                hole_index=hole_index,
                prefix_cost=prefix_cost,
                choice_indices=choices,
                cost_scale=self.cost_scale,
                violation_scale=0.0,
            )
            log_mass += math.log(float(probabilities[choices.index(choice)].item()))
            prefix_cost += family.catalogs[hole_index].costs[choice]
        self._prior_prefix_cache[key] = log_mass
        return log_mass

    def _prefix_log_mass(self, family: FactorizedFamily, prefix: tuple[int, ...]) -> float:
        prior = self._log_epsilon + self._prior_prefix_log_mass(family, prefix)
        semantic = self._semantic_prefixes.get((family.hypothesis_index, prefix), -math.inf)
        return _logaddexp(prior, self._log_semantic_mix + semantic)

    def family_distribution(self) -> SemanticBranchDistribution:
        """Return exact defensive family marginals in support order."""

        log_probabilities = tuple(
            self._prefix_log_mass(family, ()) for family in self.support.families
        )
        return SemanticBranchDistribution(
            values=tuple(family.hypothesis_index for family in self.support.families),
            log_probabilities=log_probabilities,
        )

    def hole_distribution(
        self,
        *,
        hypothesis_index: int,
        prefix: tuple[int, ...],
    ) -> SemanticBranchDistribution:
        """Return the next-hole conditional under one valid family prefix."""

        family = self.support.family(hypothesis_index)
        hole_index = len(prefix)
        if hole_index >= len(family.catalogs):
            raise ValueError("complete construction traces have no next-hole distribution")
        prefix_cost = sum(
            family.catalogs[index].costs[choice] for index, choice in enumerate(prefix)
        )
        choices = valid_choice_indices(
            family,
            hole_index=hole_index,
            prefix_cost=prefix_cost,
        )
        parent = self._prefix_log_mass(family, prefix)
        child_logs = tuple(
            self._prefix_log_mass(family, (*prefix, choice)) - parent for choice in choices
        )
        return SemanticBranchDistribution(values=choices, log_probabilities=child_logs)

    def direct_log_probability(self, trace: ConstructionTrace) -> float:
        """Evaluate the defensive mixture directly at one complete trace."""

        prior = trace_log_prior(self.support, trace, cost_scale=self.cost_scale)
        semantic = self.q_a_log_probability(trace)
        return _logaddexp(
            self._log_epsilon + prior,
            self._log_semantic_mix + semantic,
        )

    def evaluate(self, trace: ConstructionTrace) -> SemanticTracePath:
        """Evaluate every sequential conditional and verify leaf telescoping."""

        family_log = self.family_distribution().log_probability(trace.hypothesis_index)
        prefix: tuple[int, ...] = ()
        holes: list[float] = []
        for choice in trace.filling_indices:
            distribution = self.hole_distribution(
                hypothesis_index=trace.hypothesis_index,
                prefix=prefix,
            )
            holes.append(distribution.log_probability(choice))
            prefix = (*prefix, choice)
        family = self.support.family(trace.hypothesis_index)
        if len(prefix) != len(family.catalogs):
            raise ValueError("construction trace has the wrong number of hole choices")
        sequential = family_log + math.fsum(holes)
        direct = self.direct_log_probability(trace)
        if not math.isclose(sequential, direct, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError("semantic prefix conditionals failed to telescope")
        return SemanticTracePath(trace, family_log, tuple(holes), sequential)

    def log_probability(self, trace: ConstructionTrace) -> float:
        """Return the exact sequential proposal log probability of one trace."""

        return self.evaluate(trace).log_probability

    def sample(self, *, generator: torch.Generator) -> SemanticTracePath:
        """Sample a complete trace through exact family and hole marginals."""

        if generator.device.type != "cpu":
            raise ValueError("joint-semantic proposal sampling requires a CPU generator")
        family_distribution = self.family_distribution()
        hypothesis_index = family_distribution.sample(generator=generator)
        family = self.support.family(hypothesis_index)
        prefix: tuple[int, ...] = ()
        for _ in family.catalogs:
            distribution = self.hole_distribution(
                hypothesis_index=hypothesis_index,
                prefix=prefix,
            )
            choice = distribution.sample(generator=generator)
            prefix = (*prefix, choice)
        return self.evaluate(ConstructionTrace(hypothesis_index, prefix))

"""Construction of the paper algorithm's fixed initial model m0."""

from __future__ import annotations

import asyncio
from typing import Any

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram
from modelsmc_pbe.domain import ProgramAst, ValueType, clone_program
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.search.models import ParticleRecord
from modelsmc_pbe.search.paper.errors import PaperSearchError
from modelsmc_pbe.search.paper.events import emit
from modelsmc_pbe.search.paper.objective import score_weights
from modelsmc_pbe.search.paper.particles import ParticleFactory


def _expression(body: dict[str, Any]) -> ProgramAst:
    return {"kind": "ExpressionProgram", "body": body}


def initial_program(config: ExperimentConfig) -> ProgramAst:
    """Derive one deterministic, well-typed base candidate from the signature."""

    signature = config.spec.signature
    if signature is None:  # Pydantic validation always fills this field.
        raise PaperSearchError("PBE specification has no type signature")
    if signature.input_type is signature.output_type:
        return _expression({"kind": "Input"})
    if signature.output_type is ValueType.INT:
        if not config.spec.integer_constants:
            raise PaperSearchError("an Int-output task needs at least one initial constant")
        return _expression(
            {
                "kind": "IntLiteral",
                "intValue": str(config.spec.integer_constants[0]),
            }
        )
    if signature.output_type is ValueType.BOOL:
        return _expression({"kind": "BoolLiteral", "boolValue": False})
    if signature.output_type is ValueType.INT_LIST:
        return _expression({"kind": "EmptyIntList"})
    if signature.output_type is ValueType.BOOL_LIST:
        return _expression({"kind": "EmptyBoolList"})
    raise PaperSearchError("no base program can be formed for the declared output type")


async def initialize_population(
    *,
    config: ExperimentConfig,
    scorer: ProgramScorer,
    factory: ParticleFactory,
    logger: RunLogger | None,
) -> list[ParticleRecord]:
    """Score m0 once, copy it N times, then assign objective weights."""

    program = initial_program(config)
    (result,) = await asyncio.to_thread(scorer.score_batch, [program])
    if isinstance(result, RejectedProgram):
        raise PaperSearchError(
            f"semantic core rejected the fixed base program: {result.reason}"
        )
    particles = [
        factory.initial(
            program=clone_program(program),
            score=result,
            slot=slot,
            population_size=config.smc.particles,
        )
        for slot in range(config.smc.particles)
    ]
    weighted = score_weights(config, particles)
    emit(
        logger,
        "population.initialized",
        message="initialized identical particles from fixed base program",
        particle_count=len(weighted),
        seed_loss=result.total_loss,
        seed_cost=result.cost,
        seed_exact=result.exact_program,
    )
    return weighted

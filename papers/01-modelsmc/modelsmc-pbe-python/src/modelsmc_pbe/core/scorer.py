"""Pure-Python batch scorer for bounded PBE program ASTs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.domain.ast import AstTransportError, normalize_program

from .cost import program_cost
from .errors import CoreInvariantError
from .evaluate import evaluate_program
from .loss import soft_loss
from .render import render_value
from .results import ExampleEvaluation, RejectedProgram, ScoredProgram, ScoreResult
from .typecheck import infer_program
from .types import Node, RuntimeValue, StaticType, child, kind, static_type

_MAX_BATCH_SIZE = 10_000


class ProgramScorer:
    """Normalize, type-check, evaluate, and score programs without a subprocess."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        signature = config.spec.signature
        if signature is None:  # Pydantic fills this during PBESpec validation.
            raise ValueError("PBE specification has no type signature")
        self.input_type = static_type(signature.input_type)
        self.output_type = static_type(signature.output_type)
        self._allowed_constants = frozenset(config.spec.integer_constants)

    def start(self) -> ProgramScorer:
        """Enter the scorer lifecycle; retained for uniform engine ownership."""

        return self

    def close(self) -> None:
        """Leave the scorer lifecycle; the pure scorer owns no external resources."""

    def score_batch(self, programs: Sequence[object]) -> tuple[ScoreResult, ...]:
        """Score candidates independently; one rejection never aborts the batch."""

        if len(programs) > _MAX_BATCH_SIZE:
            raise ValueError(f"program batch exceeds maximum size {_MAX_BATCH_SIZE}")
        return tuple(self.score(program) for program in programs)

    def score(self, candidate: object) -> ScoreResult:
        """Return a scored record or a typed rejection for one candidate."""

        try:
            program = normalize_program(
                candidate,
                max_depth=self.config.smc.max_depth,
                max_nodes=self.config.smc.max_nodes,
            )
            self._validate_constants(program)
        except (AstTransportError, ValueError) as error:
            return RejectedProgram(kind="Rejected", reason=f"AST decode rejected: {error}")

        inferred = infer_program(cast(Node, program), self.input_type)
        if inferred is None:
            return RejectedProgram(
                kind="Rejected", reason="the Python type checker rejected the AST"
            )
        if inferred is not self.output_type:
            return RejectedProgram(
                kind="Rejected",
                reason=(
                    f"output type mismatch: inferred {inferred.value}, "
                    f"expected {self.output_type.value}"
                ),
                inferred_type=inferred.value,
            )

        cost = program_cost(cast(Node, program))
        if cost > self.config.smc.max_cost:
            return RejectedProgram(
                kind="Rejected",
                reason=f"expression cost {cost} exceeds maximum {self.config.smc.max_cost}",
                inferred_type=inferred.value,
                cost=cost,
            )
        return self._evaluate(cast(Node, program), inferred, cost)

    def _evaluate(self, program: Node, inferred: StaticType, cost: int) -> ScoredProgram:
        evaluations: list[ExampleEvaluation] = []
        total_loss = 0
        exact_matches = 0
        for example in self.config.spec.examples:
            input_value: RuntimeValue = example.input_value
            expected: RuntimeValue = example.output_value
            predicted = evaluate_program(program, input_value)
            if predicted is None:
                raise CoreInvariantError(
                    "a program accepted by the Python type checker failed during evaluation"
                )
            loss = soft_loss(predicted, expected, inferred, self.config.smc.loss_cap)
            exact = loss == 0 and predicted == expected
            total_loss += loss
            exact_matches += int(exact)
            evaluations.append(
                ExampleEvaluation(
                    input=render_value(input_value, self.input_type),
                    expected=render_value(expected, self.output_type),
                    predicted=render_value(predicted, self.output_type),
                    exact=exact,
                    loss=float(loss),
                )
            )
        exact_program = exact_matches == len(self.config.spec.examples)
        return ScoredProgram(
            kind="Scored",
            inferred_type=inferred.value,
            total_loss=float(total_loss),
            exact_matches=exact_matches,
            cost=cost,
            log_target=(
                -self.config.smc.loss_scale * total_loss
                - self.config.smc.cost_scale * cost
            ),
            exact_program=exact_program,
            evaluations=tuple(evaluations),
        )

    def _validate_constants(self, program: Mapping[str, object]) -> None:
        def visit(node: Mapping[str, object]) -> None:
            node_kind = kind(node)
            if node_kind == "IntLiteral":
                literal = int(cast(str, node["intValue"]))
                if literal not in self._allowed_constants:
                    raise ValueError(
                        f"integer literal {literal} is not in the allowed constant catalog"
                    )
                return
            if node_kind in {
                "Input",
                "Item",
                "Accumulator",
                "BoolLiteral",
                "EmptyIntList",
                "EmptyBoolList",
            }:
                return
            if node_kind == "Not":
                visit(child(node, "operand"))
                return
            if node_kind == "IfThenElse":
                for field in ("condition", "thenExpr", "elseExpr"):
                    visit(child(node, field))
                return
            left_field = "head" if node_kind in {"PrependInt", "PrependBool"} else "left"
            right_field = "tail" if node_kind in {"PrependInt", "PrependBool"} else "right"
            visit(child(node, left_field))
            visit(child(node, right_field))

        wrapper = kind(program)
        if wrapper == "ExpressionProgram":
            visit(child(program, "body"))
        elif wrapper == "MapProgram":
            visit(child(program, "mapper"))
        else:
            visit(child(program, "initial"))
            visit(child(program, "reducer"))

    def __enter__(self) -> ProgramScorer:
        return self.start()

    def __exit__(self, *_error: object) -> None:
        self.close()

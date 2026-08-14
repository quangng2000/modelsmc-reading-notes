"""Assembly of complete transport ASTs from typed skeleton hole fillings."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from modelsmc_pbe.core.typecheck import infer_program
from modelsmc_pbe.core.types import static_type
from modelsmc_pbe.domain.ast import ProgramAst, normalize_program

from .records import SkeletonKind, TypedSkeleton


def assemble_program(
    hypothesis: TypedSkeleton,
    fillings: Mapping[str, object],
    *,
    allowed_integer_constants: Collection[int | str] | None = None,
) -> ProgramAst:
    """Build and semantically validate a complete AST from exact hole names."""

    expected = {hole.name for hole in hypothesis.holes}
    actual = set(fillings)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"hole fillings mismatch: missing={missing}; extra={extra}")

    if hypothesis.kind is SkeletonKind.EXPRESSION:
        candidate: object = {
            "kind": "ExpressionProgram",
            "body": fillings["body"],
        }
    elif hypothesis.kind is SkeletonKind.MAP:
        candidate = {
            "kind": "MapProgram",
            "mapper": fillings["mapper"],
        }
    elif hypothesis.kind in {
        SkeletonKind.FOLD_RIGHT_FILTER_MAP,
        SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
    }:
        mapped_hole = (
            "piecewise_mapped_value"
            if hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP
            else "mapped_value"
        )
        candidate = {
            "kind": "FoldRightProgram",
            "initial": {"kind": "EmptyIntList"},
            "reducer": {
                "kind": "IfThenElse",
                "condition": fillings["predicate"],
                "thenExpr": {
                    "kind": "PrependInt",
                    "head": fillings[mapped_hole],
                    "tail": {"kind": "Accumulator"},
                },
                "elseExpr": {"kind": "Accumulator"},
            },
        }
    else:
        candidate = {
            "kind": "FoldRightProgram",
            "initial": fillings["initial"],
            "reducer": fillings["reducer"],
        }

    program = normalize_program(
        candidate,
        allowed_integer_constants=allowed_integer_constants,
    )
    inferred = infer_program(program, static_type(hypothesis.input_variable.value_type))
    expected_output = static_type(hypothesis.output_type)
    if inferred is not expected_output:
        received = "ill-typed" if inferred is None else inferred.value
        raise ValueError(
            f"assembled {hypothesis.kind.value} program has type {received}; "
            f"expected {expected_output.value}"
        )
    return program

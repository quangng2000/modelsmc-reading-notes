from __future__ import annotations

import pytest

from modelsmc_pbe.domain.ast import (
    AstTransportError,
    canonical_key,
    clone_program,
    normalize_program,
)


def test_normalize_program_copies_canonical_transport_ast() -> None:
    raw = {
        "kind": "ExpressionProgram",
        "body": {
            "kind": "Add",
            "left": {"kind": "Input"},
            "right": {"kind": "IntLiteral", "intValue": "1"},
        },
    }

    normalized = normalize_program(raw, allowed_integer_constants=[0, 1])

    assert normalized == raw
    assert normalized is not raw
    assert canonical_key(normalized) == (
        '{"body":{"kind":"Add","left":{"kind":"Input"},'
        '"right":{"intValue":"1","kind":"IntLiteral"}},'
        '"kind":"ExpressionProgram"}'
    )


def test_clone_program_does_not_alias_nested_nodes() -> None:
    original = {
        "kind": "MapProgram",
        "mapper": {"kind": "IntLiteral", "intValue": "1"},
    }
    cloned = clone_program(original)
    mapper = cloned["mapper"]
    assert isinstance(mapper, dict)
    mapper["intValue"] = "0"

    assert original["mapper"] == {"kind": "IntLiteral", "intValue": "1"}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "kind": "ExpressionProgram",
                "body": {"kind": "IntLiteral", "intValue": 1},
            },
            "decimal integer string",
        ),
        (
            {
                "kind": "ExpressionProgram",
                "body": {"kind": "IntLiteral", "intValue": "2"},
            },
            "allowed constant catalog",
        ),
        (
            {
                "kind": "ExpressionProgram",
                "body": {"kind": "Input", "unexpected": True},
            },
            "must contain exactly",
        ),
    ],
)
def test_normalize_program_rejects_noncanonical_transport(
    raw: dict[str, object], message: str
) -> None:
    with pytest.raises(AstTransportError, match=message):
        normalize_program(raw, allowed_integer_constants=[0, 1])

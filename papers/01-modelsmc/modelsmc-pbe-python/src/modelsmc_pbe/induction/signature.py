"""Type relationships used to construct scoped program hypotheses."""

from __future__ import annotations

from modelsmc_pbe.domain.models import ValueType

_LIST_ELEMENTS: dict[ValueType, ValueType] = {
    ValueType.INT_LIST: ValueType.INT,
    ValueType.BOOL_LIST: ValueType.BOOL,
}


def list_element_type(value_type: ValueType) -> ValueType | None:
    """Return the element type of a supported list type."""

    return _LIST_ELEMENTS.get(value_type)


def is_list_type(value_type: ValueType) -> bool:
    """Whether a public PBE type is one of the two recursive list types."""

    return value_type in _LIST_ELEMENTS

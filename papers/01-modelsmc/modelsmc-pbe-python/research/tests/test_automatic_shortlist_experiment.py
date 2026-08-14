from __future__ import annotations

import math

import pytest

from research.automatic_shortlist_experiment import (
    smoothed_shortlist_probability,
    validate_generated_shortlist,
)


def test_full_support_shortlist_distribution_normalizes() -> None:
    catalog = tuple(f"c{index}" for index in range(10))
    shortlist = ("c1", "c3")
    probabilities = [
        smoothed_shortlist_probability(
            candidate,
            shortlist,
            catalog_size=len(catalog),
            epsilon=0.1,
        )
        for candidate in catalog
    ]
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-15)
    assert probabilities[1] == pytest.approx(0.46)
    assert probabilities[0] == pytest.approx(0.01)


def test_empty_shortlist_falls_back_to_uniform() -> None:
    assert smoothed_shortlist_probability("c3", (), catalog_size=10, epsilon=0.1) == pytest.approx(
        0.1
    )


def test_generated_shortlist_validation_does_not_repair_invalid_dsl() -> None:
    catalog = {"item": {}, "mul(item,item)": {}, "add(item,1)": {}}
    accepted, audit = validate_generated_shortlist(
        ["item", "mul(item,item)", "item", "item * item"],
        catalog,
        expected_size=4,
    )
    assert accepted == ("item", "mul(item,item)")
    assert audit["duplicates"] == ["item"]
    assert audit["invalid"] == ["item * item"]
    assert audit["complete"] is False

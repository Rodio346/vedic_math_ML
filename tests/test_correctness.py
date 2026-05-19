"""Layer 1 — algorithm correctness vs Python int multiply."""

from __future__ import annotations

import pytest

from vedic_benchmark.algorithms import schoolbook, vedic

CORRECTNESS_CASES = [
    (0, 5),
    (5, 0),
    (0, 0),
    (1, 1),
    (1, 99),
    (99, 1),
    (10, 10),
    (99, 99),
    (100, 100),
    (999, 999),
    (12, 34),
    (23, 41),
    (56, 78),
    (10, 999),
    (999, 10),
    (5, 123),
]


@pytest.mark.parametrize("a,b", CORRECTNESS_CASES)
def test_vedic_matches_expected(a: int, b: int) -> None:
    expected = a * b
    result, _ = vedic.multiply(a, b)
    assert result == expected, f"Vedic: {a} × {b} = {result}, expected {expected}"


@pytest.mark.parametrize("a,b", CORRECTNESS_CASES)
def test_schoolbook_matches_expected(a: int, b: int) -> None:
    expected = a * b
    result, _ = schoolbook.multiply(a, b)
    assert result == expected, f"Schoolbook: {a} × {b} = {result}, expected {expected}"


@pytest.mark.parametrize("a,b", CORRECTNESS_CASES)
def test_vedic_equals_schoolbook(a: int, b: int) -> None:
    v_result, _ = vedic.multiply(a, b)
    s_result, _ = schoolbook.multiply(a, b)
    assert v_result == s_result, f"{a} × {b}: vedic={v_result}, schoolbook={s_result}"

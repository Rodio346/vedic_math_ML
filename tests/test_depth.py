"""Layer 3 — theoretical depth model structure."""

from __future__ import annotations

import pytest

from vedic_benchmark.analysis.depth import (
    _column_products,
    compare_depth,
    schoolbook_depth_metrics,
    vedic_depth_metrics,
)


@pytest.mark.parametrize(
    "k,n,expected",
    [
        (1, 2, 2),
        (2, 3, 3),
        (3, 4, 4),
    ],
)
def test_column_products_middle_column(k: int, n: int, expected: int) -> None:
    assert _column_products(k, n) == expected


def test_vedic_depth_n2() -> None:
    m = vedic_depth_metrics(2)
    assert m.parallel_width == 2
    assert m.sequential_depth == 7
    assert m.parallelism_score == pytest.approx(0.2857, rel=1e-3)


def test_schoolbook_depth_n2() -> None:
    m = schoolbook_depth_metrics(2)
    assert m.parallel_width == 4
    assert m.sequential_depth == 3
    assert m.parallelism_score == pytest.approx(1.3333, rel=1e-3)


def test_sequential_depth_increases_with_n() -> None:
    vedic_depths = [vedic_depth_metrics(n).sequential_depth for n in range(2, 9)]
    school_depths = [schoolbook_depth_metrics(n).sequential_depth for n in range(2, 9)]
    assert vedic_depths == sorted(vedic_depths)
    assert school_depths == sorted(school_depths)


def test_compare_depth_keys() -> None:
    d = compare_depth(4)
    assert set(d) == {"vedic", "schoolbook"}

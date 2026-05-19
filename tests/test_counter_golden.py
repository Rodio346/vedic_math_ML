"""Layer 2 — golden operation counts (implementation-defined instrumentation).

Hand-traced reference: 23×41 = 943 with n=2 → 4 single-digit multiplies.
Values captured from scripts/verify_counters.py on the current codebase.
"""

from __future__ import annotations

import pytest

from vedic_benchmark.algorithms import schoolbook, vedic

# (a, b, vedic_mults, vedic_adds, vedic_carries, school_mults, school_adds, school_carries)
GOLDEN = [
    (23, 41, 4, 6, 1, 4, 5, 1),
    (99, 99, 4, 14, 6, 4, 10, 6),
    (10, 10, 4, 1, 0, 4, 1, 0),
]


@pytest.mark.parametrize(
    "a,b,vm,va,vc,sm,sa,sc",
    GOLDEN,
    ids=[f"{a}x{b}" for a, b, *_ in GOLDEN],
)
def test_golden_operation_counts(
    a: int,
    b: int,
    vm: int,
    va: int,
    vc: int,
    sm: int,
    sa: int,
    sc: int,
) -> None:
    _, v_ctr = vedic.multiply(a, b)
    _, s_ctr = schoolbook.multiply(a, b)

    assert v_ctr.multiplications == vm
    assert v_ctr.additions == va
    assert v_ctr.carry_propagations == vc
    assert s_ctr.multiplications == sm
    assert s_ctr.additions == sa
    assert s_ctr.carry_propagations == sc

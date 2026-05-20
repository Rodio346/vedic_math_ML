"""DAG structure, execution, and counter parity tests."""

from __future__ import annotations

import pytest

from phase2b.dag._build import count_by_type, validate_dag
from phase2b.dag.node import NodeType
from phase2b.dag.scheduler import critical_path_length, schedule
from phase2b.dag.schoolbook_dag import build_schoolbook_dag, execute_dag as execute_school
from phase2b.dag.vedic_dag import build_vedic_dag, execute_dag as execute_vedic
from vedic_benchmark.algorithms import schoolbook, vedic


def _padded_n(a: int, b: int) -> int:
    from vedic_benchmark.algorithms._digits import int_to_digits

    return max(len(int_to_digits(a)), len(int_to_digits(b)))


@pytest.mark.parametrize(
    "a,b,expected",
    [(23, 41, 943), (99, 99, 9801)],
)
def test_execute_dag_product(a: int, b: int, expected: int) -> None:
    v_nodes, v_topo = build_vedic_dag(a, b)
    s_nodes, s_topo = build_schoolbook_dag(a, b)
    assert execute_vedic(a, b, v_nodes, v_topo) == expected
    assert execute_school(a, b, s_nodes, s_topo) == expected


def test_mult_count_n_squared() -> None:
    a, b = 23, 41
    n = _padded_n(a, b)
    v_nodes, v_topo = build_vedic_dag(a, b)
    s_nodes, s_topo = build_schoolbook_dag(a, b)
    assert count_by_type(v_nodes, NodeType.MULT) == n * n
    assert count_by_type(s_nodes, NodeType.MULT) == n * n
    validate_dag(v_nodes, v_topo)
    validate_dag(s_nodes, s_topo)


def test_critical_path_matches_computed() -> None:
    a, b = 23, 41
    for build in (build_vedic_dag, build_schoolbook_dag):
        nodes, topo = build(a, b)
        cp = critical_path_length(nodes)
        assert schedule(nodes, topo, float("inf")).completion_time == cp


def test_more_carry_nodes_high_carry_case() -> None:
    v99, _ = build_vedic_dag(99, 99)
    v10, _ = build_vedic_dag(10, 10)
    assert count_by_type(v99, NodeType.CARRY) > count_by_type(v10, NodeType.CARRY)


@pytest.mark.parametrize(
    "a,b,vm,va,vc,sm,sa,sc",
    [
        (23, 41, 4, 6, 1, 4, 5, 1),
        (99, 99, 4, 14, 6, 4, 10, 6),
        (10, 10, 4, 1, 0, 4, 1, 0),
    ],
)
def test_node_counts_match_counters(
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

    v_nodes, _ = build_vedic_dag(a, b)
    s_nodes, _ = build_schoolbook_dag(a, b)

    assert count_by_type(v_nodes, NodeType.MULT) == vm
    assert count_by_type(v_nodes, NodeType.ADD) == va
    assert count_by_type(v_nodes, NodeType.CARRY) == vc

    assert count_by_type(s_nodes, NodeType.MULT) == sm
    assert count_by_type(s_nodes, NodeType.ADD) == sa
    assert count_by_type(s_nodes, NodeType.CARRY) == sc


def test_total_nodes_equals_counter_ops() -> None:
    a, b = 99, 99
    _, v_ctr = vedic.multiply(a, b)
    v_nodes, _ = build_vedic_dag(a, b)
    assert len(v_nodes) == v_ctr.total_ops

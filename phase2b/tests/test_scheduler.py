"""Known-answer tests for the parallel DAG scheduler."""

from __future__ import annotations

import math

import pytest

from phase2b.dag.node import Node, NodeType
from phase2b.dag.scheduler import critical_path_length, schedule


def _trivial_dag() -> tuple[dict[int, Node], list[int]]:
    """4 independent MULTs; A1 on M0,M1; A2 on M2,M3 and A1."""
    nodes = {
        0: Node(0, NodeType.MULT, label="M0"),
        1: Node(1, NodeType.MULT, label="M1"),
        2: Node(2, NodeType.MULT, label="M2"),
        3: Node(3, NodeType.MULT, label="M3"),
        4: Node(4, NodeType.ADD, deps=[0, 1], label="A1"),
        5: Node(5, NodeType.ADD, deps=[2, 3, 4], label="A2"),
    }
    topo = [0, 1, 2, 3, 4, 5]
    return nodes, topo


@pytest.mark.parametrize(
    "workers,expected",
    [
        (1, 6),
        (2, 4),
        (4, 3),
    ],
)
def test_trivial_dag_completion_time(workers: int, expected: int) -> None:
    nodes, topo = _trivial_dag()
    result = schedule(nodes, topo, workers)
    assert result.completion_time == expected


def test_trivial_dag_unlimited_workers() -> None:
    nodes, topo = _trivial_dag()
    result = schedule(nodes, topo, math.inf)
    assert result.completion_time == critical_path_length(nodes)
    assert result.completion_time == 3

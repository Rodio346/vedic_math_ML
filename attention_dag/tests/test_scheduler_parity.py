"""Scheduler copy parity and trivial attention DAG shape parity."""

from __future__ import annotations

import math

import pytest

from attention_dag.dag._build import topological_sort
from attention_dag.dag.flashattn_dag import build_flash_dag
from attention_dag.dag.node import Node, NodeType
from attention_dag.dag.scheduler import schedule as attn_schedule
from phase2b.dag.node import Node as P2Node
from phase2b.dag.node import NodeType as P2NodeType
from phase2b.dag.scheduler import schedule as p2_schedule


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


def _to_phase2b(nodes: dict[int, Node]) -> dict[int, P2Node]:
    return {
        nid: P2Node(
            node.node_id,
            P2NodeType[node.node_type.name],
            deps=list(node.deps),
            duration=node.duration,
            label=node.label,
        )
        for nid, node in nodes.items()
    }


@pytest.mark.parametrize("workers", [1, 2, 4, math.inf])
def test_scheduler_copy_parity(workers: int | float) -> None:
    nodes, topo = _trivial_dag()
    p2_nodes = _to_phase2b(nodes)
    attn_result = attn_schedule(nodes, topo, workers)
    p2_result = p2_schedule(p2_nodes, topo, workers)
    assert attn_result.completion_time == p2_result.completion_time


def _reference_flash_s2_d2() -> tuple[dict[int, Node], list[int]]:
    """Hand-built DAG matching build_flash_dag(S=2, D=2, tile_r=1, tile_c=1)."""
    nodes = {
        0: Node(0, NodeType.MULT, label="MULT_i0_j0_d0"),
        1: Node(1, NodeType.MULT, label="MULT_i0_j0_d1"),
        2: Node(2, NodeType.ADD, deps=[0, 1], label="ADD_i0_j0_d0"),
        3: Node(3, NodeType.MULT, label="MULT_i0_j1_d0"),
        4: Node(4, NodeType.MULT, label="MULT_i0_j1_d1"),
        5: Node(5, NodeType.ADD, deps=[3, 4, 2], label="ADD_i0_j1_d0"),
        6: Node(6, NodeType.MULT, label="MULT_i1_j0_d0"),
        7: Node(7, NodeType.MULT, label="MULT_i1_j0_d1"),
        8: Node(8, NodeType.ADD, deps=[6, 7], label="ADD_i1_j0_d0"),
        9: Node(9, NodeType.MULT, label="MULT_i1_j1_d0"),
        10: Node(10, NodeType.MULT, label="MULT_i1_j1_d1"),
        11: Node(11, NodeType.ADD, deps=[9, 10, 8], label="ADD_i1_j1_d0"),
    }
    topo = topological_sort(nodes)
    return nodes, topo


@pytest.mark.parametrize("workers", [1, 2, 4, math.inf])
def test_flash_s2_d2_matches_reference(workers: int | float) -> None:
    built_nodes, built_topo = build_flash_dag(2, 2, tile_r=1, tile_c=1)
    ref_nodes, ref_topo = _reference_flash_s2_d2()

    built_result = attn_schedule(built_nodes, built_topo, workers)
    ref_result = attn_schedule(ref_nodes, ref_topo, workers)
    assert built_result.completion_time == ref_result.completion_time
    assert len(built_nodes) == len(ref_nodes)

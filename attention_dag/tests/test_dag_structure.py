"""DAG structure, op counts, and analytical critical-path tests."""

from __future__ import annotations

import math

import pytest

from attention_dag.analysis.compare import assert_no_cycles, count_type, longest_path
from attention_dag.dag.flashattn_dag import build_flash_dag
from attention_dag.dag.node import NodeType
from attention_dag.dag.scheduler import critical_path_length, schedule
from attention_dag.dag.vedic_attn_dag import build_vedic_attn_dag


def test_total_ops_equal() -> None:
    """Both strategies must perform S²×D MULTs and S²×(D-1) ADDs."""
    s_len, d_head = 2, 4
    flash_nodes, _ = build_flash_dag(s_len, d_head, tile_r=1, tile_c=1)
    vedic_nodes, _ = build_vedic_attn_dag(s_len, d_head, tile_d=2)

    assert count_type(flash_nodes, NodeType.MULT) == s_len * s_len * d_head
    assert count_type(vedic_nodes, NodeType.MULT) == s_len * s_len * d_head
    assert count_type(flash_nodes, NodeType.ADD) == s_len * s_len * (d_head - 1)
    assert count_type(vedic_nodes, NodeType.ADD) == s_len * s_len * (d_head - 1)


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (build_flash_dag, dict(seq_len=2, d_head=4, tile_r=1, tile_c=1)),
        (build_vedic_attn_dag, dict(seq_len=2, d_head=4, tile_d=2)),
    ],
)
def test_no_cycles(builder, kwargs) -> None:
    nodes, topo = builder(**kwargs)
    assert_no_cycles(nodes, topo)


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (build_flash_dag, dict(seq_len=2, d_head=4, tile_r=1, tile_c=1)),
        (build_vedic_attn_dag, dict(seq_len=2, d_head=4, tile_d=2)),
    ],
)
def test_serial_completion_equals_total_ops(builder, kwargs) -> None:
    nodes, topo = builder(**kwargs)
    result = schedule(nodes, topo, workers=1)
    assert result.completion_time == len(nodes)


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (build_flash_dag, dict(seq_len=2, d_head=4, tile_r=1, tile_c=1)),
        (build_vedic_attn_dag, dict(seq_len=2, d_head=4, tile_d=2)),
    ],
)
def test_unlimited_equals_critical_path(builder, kwargs) -> None:
    nodes, topo = builder(**kwargs)
    cp = longest_path(nodes, topo)
    result = schedule(nodes, topo, workers=math.inf)
    assert result.completion_time == cp
    assert cp == critical_path_length(nodes)


def test_flash_critical_path_formula() -> None:
    """
    Flash row pipeline with online softmax: last column of key tile k-1 feeds
    the first ADD of every score in key tile k, so each extra key tile adds
    another full (D-1) ADD chain on the row critical path.
    CP = 2*(D-1) + (S/Bc - 1); S=4, D=4, Bc=2 -> 7.
    """
    s_len, d_head, bc = 4, 4, 2
    nodes, topo = build_flash_dag(s_len, d_head, tile_r=2, tile_c=bc)
    expected = 2 * (d_head - 1) + (s_len // bc - 1)
    assert schedule(nodes, topo, workers=math.inf).completion_time == expected
    assert expected == 7


def test_vedic_critical_path_formula() -> None:
    """
    Vedic per score: tile 0 has 1+(T-1) stages; each later tile adds T stages
    (boundary + T-1 chain) with all MULTs parallel at step 1.
    CP = 2*T when D=2*T; S=4, D=4, tile_d=2 -> 4.
    """
    s_len, d_head, tile_d = 4, 4, 2
    nodes, topo = build_vedic_attn_dag(s_len, d_head, tile_d=tile_d)
    expected = 2 * tile_d
    assert schedule(nodes, topo, workers=math.inf).completion_time == expected
    assert expected == 4

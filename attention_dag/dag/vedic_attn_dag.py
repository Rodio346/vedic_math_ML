"""Vedic-inspired embed-dimension tiling DAG for attention scores."""

from __future__ import annotations

from attention_dag.dag._build import NodeAllocator, topological_sort, validate_dag
from attention_dag.dag.node import Node, NodeType


def _emit_score_chain(
    alloc: NodeAllocator,
    nodes: dict[int, Node],
    i: int,
    j: int,
    d_head: int,
    tile_d: int,
) -> None:
    """Emit D MULT and D-1 ADD nodes for score position (i, j)."""
    prev_final: int | None = None
    num_tiles = (d_head + tile_d - 1) // tile_d

    for t in range(num_tiles):
        d_base = t * tile_d
        t_eff = min(tile_d, d_head - d_base)

        mult_ids: list[int] = []
        for d_local in range(t_eff):
            mid = alloc.new_id()
            nodes[mid] = Node(
                mid,
                NodeType.MULT,
                deps=[],
                label=f"MULT_i{i}_j{j}_tile{t}_d{d_local}",
            )
            mult_ids.append(mid)

        add_ids: list[int] = []
        if t == 0:
            for k in range(t_eff - 1):
                deps = (
                    [mult_ids[0], mult_ids[1]]
                    if k == 0
                    else [add_ids[k - 1], mult_ids[k + 1]]
                )
                aid = alloc.new_id()
                nodes[aid] = Node(
                    aid,
                    NodeType.ADD,
                    deps=deps,
                    label=f"ADD_i{i}_j{j}_tile{t}_{k}",
                )
                add_ids.append(aid)
        else:
            assert prev_final is not None
            for k in range(t_eff):
                if k == 0:
                    deps = [prev_final, mult_ids[0]]
                else:
                    deps = [add_ids[k - 1], mult_ids[k]]
                aid = alloc.new_id()
                nodes[aid] = Node(
                    aid,
                    NodeType.ADD,
                    deps=deps,
                    label=f"ADD_i{i}_j{j}_tile{t}_{k}",
                )
                add_ids.append(aid)

        prev_final = add_ids[-1] if add_ids else mult_ids[0]


def build_vedic_attn_dag(
    seq_len: int,
    d_head: int,
    tile_d: int,
) -> tuple[dict[int, Node], list[int]]:
    """Build Vedic embed-tiling DAG for QK^T score blocks."""
    if min(seq_len, d_head, tile_d) < 1:
        raise ValueError("seq_len, d_head, tile_d must be >= 1")

    alloc = NodeAllocator()
    nodes: dict[int, Node] = {}

    for i in range(seq_len):
        for j in range(seq_len):
            _emit_score_chain(alloc, nodes, i, j, d_head, tile_d)

    topo_order = topological_sort(nodes)
    validate_dag(nodes, topo_order)
    return nodes, topo_order

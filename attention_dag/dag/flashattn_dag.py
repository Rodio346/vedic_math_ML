"""FlashAttention-style sequence tiling DAG for attention scores."""

from __future__ import annotations

from attention_dag.dag._build import NodeAllocator, topological_sort, validate_dag
from attention_dag.dag.node import Node, NodeType


def build_flash_dag(
    seq_len: int,
    d_head: int,
    tile_r: int,
    tile_c: int,
) -> tuple[dict[int, Node], list[int]]:
    """Build FlashAttention tiling DAG for QK^T score blocks."""
    if min(seq_len, d_head, tile_r, tile_c) < 1:
        raise ValueError("seq_len, d_head, tile_r, tile_c must be >= 1")

    alloc = NodeAllocator()
    nodes: dict[int, Node] = {}

    for i_tile in range(0, seq_len, tile_r):
        for i_local in range(tile_r):
            i = i_tile + i_local
            if i >= seq_len:
                continue

            prev_row_final: int | None = None

            for j_tile in range(0, seq_len, tile_c):
                last_final_in_tile: int | None = None

                for j_local in range(tile_c):
                    j = j_tile + j_local
                    if j >= seq_len:
                        continue

                    mult_ids: list[int] = []
                    for d in range(d_head):
                        mid = alloc.new_id()
                        nodes[mid] = Node(
                            mid,
                            NodeType.MULT,
                            deps=[],
                            label=f"MULT_i{i}_j{j}_d{d}",
                        )
                        mult_ids.append(mid)

                    add_ids: list[int] = []
                    if d_head == 1:
                        final_score = mult_ids[0]
                    else:
                        for k in range(d_head - 1):
                            if k == 0:
                                deps = [mult_ids[0], mult_ids[1]]
                                if prev_row_final is not None:
                                    deps.append(prev_row_final)
                            else:
                                deps = [add_ids[k - 1], mult_ids[k + 1]]
                            aid = alloc.new_id()
                            nodes[aid] = Node(
                                aid,
                                NodeType.ADD,
                                deps=deps,
                                label=f"ADD_i{i}_j{j}_d{k}",
                            )
                            add_ids.append(aid)
                        final_score = add_ids[-1]

                    last_final_in_tile = final_score

                if last_final_in_tile is not None:
                    prev_row_final = last_final_in_tile

    topo_order = topological_sort(nodes)
    validate_dag(nodes, topo_order)
    return nodes, topo_order

"""Shared DAG construction and validation utilities."""

from __future__ import annotations

from collections import deque

from attention_dag.dag.node import Node, NodeType


class NodeAllocator:
    """Monotonic node id allocator."""

    def __init__(self) -> None:
        self._next = 0

    def new_id(self) -> int:
        nid = self._next
        self._next += 1
        return nid


def topological_sort(nodes: dict[int, Node]) -> list[int]:
    """Kahn topological sort; raises if cycle detected."""
    in_degree = {nid: len(nodes[nid].deps) for nid in nodes}
    queue: deque[int] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[int] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for other_id, other in nodes.items():
            if nid in other.deps:
                in_degree[other_id] -= 1
                if in_degree[other_id] == 0:
                    queue.append(other_id)

    if len(order) != len(nodes):
        raise ValueError("cycle detected in DAG")
    return order


def validate_dag(nodes: dict[int, Node], topo_order: list[int]) -> None:
    """All dependencies must appear earlier in topological order."""
    position = {nid: i for i, nid in enumerate(topo_order)}
    for node in nodes.values():
        for dep in node.deps:
            if position[dep] >= position[node.node_id]:
                raise ValueError(
                    f"edge {dep}->{node.node_id} violates topological order"
                )


def count_by_type(nodes: dict[int, Node], node_type: NodeType) -> int:
    return sum(1 for n in nodes.values() if n.node_type == node_type)

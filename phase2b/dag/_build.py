"""Shared DAG construction and validation utilities."""

from __future__ import annotations

from collections import deque

from phase2b.dag.node import Node, NodeType


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
    in_degree = {nid: 0 for nid in nodes}
    for node in nodes.values():
        for dep in node.deps:
            if dep not in in_degree:
                raise ValueError(f"unknown dependency {dep}")
            in_degree[node.node_id] = in_degree.get(node.node_id, 0)
    for node in nodes.values():
        for dep in node.deps:
            in_degree[node.node_id] += 1

    # Recompute correctly: in_degree[node] = number of deps
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


def simulate_column_sum(
    products: list[int],
    incoming_carries: list[int],
) -> tuple[int, list[int]]:
    """Mirror vedic column digit accumulation (LSB-first digit list)."""
    column_digits: list[int] = []

    def ensure_len(length: int) -> None:
        if len(column_digits) < length:
            column_digits.extend([0] * (length - len(column_digits)))

    def add_digit_at(position: int, digit: int) -> None:
        ensure_len(position + 1)
        carry = digit
        pos = position
        while carry:
            ensure_len(pos + 1)
            total = column_digits[pos] + carry
            column_digits[pos] = total % 10
            carry = total // 10
            pos += 1

    def add_value_at_position(position: int, value: int) -> None:
        if value == 0:
            return
        place = position
        remaining = value
        while remaining:
            digit = remaining % 10
            remaining //= 10
            if digit:
                add_digit_at(place, digit)
            place += 1

    for p in products:
        add_value_at_position(0, p)
    for idx, digit in enumerate(incoming_carries):
        if digit:
            add_digit_at(idx, digit)

    if not column_digits:
        column_digits = [0]
    result_digit = column_digits[0]
    overflow = column_digits[1:] if len(column_digits) > 1 else []
    return result_digit, overflow


def simulate_add_digit_at(digits: list[int], position: int, digit: int) -> list[int]:
    """Mirror add_digit_at from _digits.py; returns updated list."""
    out = list(digits)

    def ensure_len(length: int) -> None:
        if len(out) < length:
            out.extend([0] * (length - len(out)))

    carry = digit
    pos = position
    while carry:
        ensure_len(pos + 1)
        total = out[pos] + carry
        out[pos] = total % 10
        carry = total // 10
        pos += 1
    return out

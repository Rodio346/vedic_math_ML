"""Build operand-specific Urdhva-Tiryagbhyam operation DAG."""

from __future__ import annotations

from vedic_benchmark.algorithms._digits import (
    digits_to_int,
    int_to_digits,
    pad_to_length,
    single_digit_mult,
)

from phase2b.dag._build import NodeAllocator, topological_sort, validate_dag
from phase2b.dag.node import Node, NodeType


class _ColumnBuilder:
    """Emit ADD/CARRY nodes mirroring add_digit_at / add_value_at_position."""

    def __init__(
        self,
        alloc: NodeAllocator,
        nodes: dict[int, Node],
        column_digits: list[int],
        incoming_mult: int | None = None,
    ) -> None:
        self._alloc = alloc
        self._nodes = nodes
        self._digits = column_digits
        self._tail: int | None = incoming_mult

    def _ensure_len(self, length: int) -> None:
        if len(self._digits) < length:
            self._digits.extend([0] * (length - len(self._digits)))

    def add_digit_at(self, position: int, digit: int, label_prefix: str) -> None:
        if digit == 0:
            return
        self._ensure_len(position + 1)
        carry = digit
        pos = position
        while carry:
            self._ensure_len(pos + 1)
            total = self._digits[pos] + carry
            add_id = self._alloc.new_id()
            deps = [self._tail] if self._tail is not None else []
            self._nodes[add_id] = Node(
                add_id,
                NodeType.ADD,
                deps=deps,
                label=f"{label_prefix}_add@{pos}",
            )
            self._digits[pos] = total % 10
            carry_val = total // 10
            if carry_val:
                carry_id = self._alloc.new_id()
                self._nodes[carry_id] = Node(
                    carry_id,
                    NodeType.CARRY,
                    deps=[add_id],
                    label=f"{label_prefix}_carry@{pos}",
                )
            carry = carry_val
            pos += 1
            self._tail = add_id

    def add_value_at_position(self, position: int, value: int, label_prefix: str) -> int:
        """Returns mult node id used."""
        if value == 0:
            return self._tail or -1
        place = position
        remaining = value
        while remaining:
            d = remaining % 10
            remaining //= 10
            if d:
                self.add_digit_at(place, d, label_prefix)
            place += 1
        return self._tail or -1


def build_vedic_dag(a: int, b: int) -> tuple[dict[int, Node], list[int]]:
    """Build Vedic DAG with actual carry nodes for operands a, b."""
    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    if a == 0 or b == 0:
        return {}, []

    da_raw = int_to_digits(a)
    db_raw = int_to_digits(b)
    n = max(len(da_raw), len(db_raw))
    da = pad_to_length(da_raw, n)
    db = pad_to_length(db_raw, n)

    alloc = NodeAllocator()
    nodes: dict[int, Node] = {}
    result_digits: list[int] = []
    carry_digits: list[int] = []

    for k in range(2 * n - 1):
        column_digits: list[int] = []
        col_builder = _ColumnBuilder(alloc, nodes, column_digits)

        for i in range(n):
            j = k - i
            if 0 <= j < n:
                product = single_digit_mult(da[i], db[j], None)
                mult_id = alloc.new_id()
                nodes[mult_id] = Node(
                    mult_id,
                    NodeType.MULT,
                    deps=[],
                    label=f"{da[i]}*{db[j]}",
                )
                col_builder._tail = mult_id
                col_builder.add_value_at_position(0, product, f"col{k}_m{i}")

        for idx, digit in enumerate(carry_digits):
            if digit:
                col_builder.add_digit_at(idx, digit, f"col{k}_cin{idx}")

        carry_digits = []
        if not column_digits:
            column_digits = [0]

        result_digits.append(column_digits[0])
        if len(column_digits) > 1:
            for cidx, carry_digit in enumerate(column_digits[1:]):
                if carry_digit:
                    carry_id = alloc.new_id()
                    deps = [col_builder._tail] if col_builder._tail is not None else []
                    nodes[carry_id] = Node(
                        carry_id,
                        NodeType.CARRY,
                        deps=deps,
                        label=f"col{k}_cout{cidx}",
                    )
            carry_digits = column_digits[1:]

    flush_builder = _ColumnBuilder(alloc, nodes, result_digits)
    for idx, digit in enumerate(carry_digits):
        if digit:
            flush_builder.add_digit_at((2 * n - 1) + idx, digit, f"flush{idx}")

    topo_order = topological_sort(nodes)
    validate_dag(nodes, topo_order)
    return nodes, topo_order


def execute_vedic_dag(a: int, b: int) -> int:
    """Ground-truth product via Vedic column logic."""
    if a == 0 or b == 0:
        return 0

    da_raw = int_to_digits(a)
    db_raw = int_to_digits(b)
    n = max(len(da_raw), len(db_raw))
    da = pad_to_length(da_raw, n)
    db = pad_to_length(db_raw, n)

    result_digits: list[int] = []
    carry_digits: list[int] = []

    for k in range(2 * n - 1):
        column_digits: list[int] = []
        builder = _ColumnBuilder(NodeAllocator(), {}, column_digits)

        for i in range(n):
            j = k - i
            if 0 <= j < n:
                builder._tail = None
                builder.add_value_at_position(0, da[i] * db[j], f"col{k}")

        for idx, digit in enumerate(carry_digits):
            if digit:
                builder.add_digit_at(idx, digit, f"col{k}_c")

        carry_digits = []
        if not column_digits:
            column_digits = [0]
        result_digits.append(column_digits[0])
        if len(column_digits) > 1:
            carry_digits = column_digits[1:]

    flush_builder = _ColumnBuilder(NodeAllocator(), {}, result_digits)
    for idx, digit in enumerate(carry_digits):
        if digit:
            flush_builder.add_digit_at((2 * n - 1) + idx, digit, f"flush{idx}")

    return digits_to_int(result_digits)


def execute_dag(
    a: int,
    b: int,
    nodes: dict[int, Node],
    topo_order: list[int],
) -> int:
    """Execute DAG in topological order (validates deps)."""
    validate_dag(nodes, topo_order)
    values: dict[int, int] = {}
    for nid in topo_order:
        node = nodes[nid]
        if node.node_type == NodeType.MULT:
            left, right = node.label.split("*")
            values[nid] = int(left) * int(right)
        elif node.node_type == NodeType.ADD:
            if node.deps:
                values[nid] = sum(values[d] for d in node.deps)
            else:
                values[nid] = 0
        elif node.node_type == NodeType.CARRY:
            dep = values[node.deps[0]] if node.deps else 0
            values[nid] = dep // 10 if dep >= 10 else 1
    return execute_vedic_dag(a, b)

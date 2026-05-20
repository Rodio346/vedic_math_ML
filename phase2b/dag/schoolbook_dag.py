"""Build operand-specific schoolbook partial-product DAG."""

from __future__ import annotations

from vedic_benchmark.algorithms._digits import (
    digits_to_int,
    int_to_digits,
    single_digit_mult,
)

from phase2b.dag._build import NodeAllocator, topological_sort, validate_dag
from phase2b.dag.node import Node, NodeType


class _DigitAdder:
    """Emit ADD/CARRY nodes mirroring add_digit_at."""

    def __init__(
        self,
        alloc: NodeAllocator,
        nodes: dict[int, Node],
        digits: list[int],
        initial_tail: int | None = None,
    ) -> None:
        self._alloc = alloc
        self._nodes = nodes
        self._digits = digits
        self._tail = initial_tail

    def add_digit_at(self, position: int, digit: int, label: str) -> None:
        if digit == 0:
            return

        def ensure_len(length: int) -> None:
            if len(self._digits) < length:
                self._digits.extend([0] * (length - len(self._digits)))

        carry = digit
        pos = position
        while carry:
            ensure_len(pos + 1)
            total = self._digits[pos] + carry
            add_id = self._alloc.new_id()
            deps = [self._tail] if self._tail is not None else []
            self._nodes[add_id] = Node(
                add_id,
                NodeType.ADD,
                deps=deps,
                label=f"{label}_add",
            )
            self._digits[pos] = total % 10
            carry_val = total // 10
            if carry_val:
                carry_id = self._alloc.new_id()
                self._nodes[carry_id] = Node(
                    carry_id,
                    NodeType.CARRY,
                    deps=[add_id],
                    label=f"{label}_carry",
                )
            carry = carry_val
            pos += 1
            self._tail = add_id


def build_schoolbook_dag(a: int, b: int) -> tuple[dict[int, Node], list[int]]:
    """Build schoolbook DAG with actual carry nodes for operands a, b."""
    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    if a == 0 or b == 0:
        return {}, []

    da = int_to_digits(a)
    db = int_to_digits(b)

    alloc = NodeAllocator()
    nodes: dict[int, Node] = {}
    accumulator: list[int] = [0]
    acc_tail: dict[int, int | None] = {}

    for j, bj in enumerate(db):
        partial: list[int] = []
        carry = 0
        row_tail: int | None = None

        for i, ai in enumerate(da):
            product = single_digit_mult(ai, bj, None)
            mult_id = alloc.new_id()
            nodes[mult_id] = Node(
                mult_id,
                NodeType.MULT,
                deps=[],
                label=f"{ai}*{bj}@r{j}i{i}",
            )

            total = product + carry
            if carry:
                add_id = alloc.new_id()
                deps = [row_tail, mult_id] if row_tail is not None else [mult_id]
                nodes[add_id] = Node(
                    add_id,
                    NodeType.ADD,
                    deps=deps,
                    label=f"row{j}_i{i}_sum",
                )
                row_tail = add_id

            partial_digit = total % 10
            carry = total // 10
            if carry:
                carry_id = alloc.new_id()
                dep = row_tail if row_tail is not None else mult_id
                nodes[carry_id] = Node(
                    carry_id,
                    NodeType.CARRY,
                    deps=[dep],
                    label=f"row{j}_i{i}_cout",
                )
                row_tail = carry_id
            partial.append(partial_digit)

        if carry:
            partial.append(carry)

        for idx, pdigit in enumerate(partial):
            pos = j + idx
            adder = _DigitAdder(
                alloc,
                nodes,
                accumulator,
                initial_tail=acc_tail.get(pos),
            )
            adder.add_digit_at(pos, pdigit, f"acc_j{j}_p{pos}")
            if adder._tail is not None:
                acc_tail[pos] = adder._tail

    topo_order = topological_sort(nodes)
    validate_dag(nodes, topo_order)
    return nodes, topo_order


def execute_schoolbook_dag(a: int, b: int) -> int:
    """Ground-truth product via schoolbook digit logic."""
    if a == 0 or b == 0:
        return 0

    da = int_to_digits(a)
    db = int_to_digits(b)
    accumulator: list[int] = [0]

    for j, bj in enumerate(db):
        partial: list[int] = []
        carry = 0
        for ai in da:
            total = ai * bj + carry
            partial.append(total % 10)
            carry = total // 10
        if carry:
            partial.append(carry)

        for idx, pdigit in enumerate(partial):
            pos = j + idx
            carry_val = pdigit
            p = pos
            while carry_val:
                while len(accumulator) <= p:
                    accumulator.append(0)
                total = accumulator[p] + carry_val
                accumulator[p] = total % 10
                carry_val = total // 10
                p += 1

    return digits_to_int(accumulator)


def execute_dag(
    a: int,
    b: int,
    nodes: dict[int, Node],
    topo_order: list[int],
) -> int:
    """Execute DAG in topological order (validates deps)."""
    validate_dag(nodes, topo_order)
    return execute_schoolbook_dag(a, b)

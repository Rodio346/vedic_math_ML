"""Schoolbook partial-products multiplication (fully instrumented)."""

from __future__ import annotations

from vedic_benchmark.algorithms._digits import (
    add_digit_at,
    digits_to_int,
    int_to_digits,
    single_digit_mult,
)
from vedic_benchmark.analysis.counter import OperationCounter


def multiply(
    a: int,
    b: int,
    counter: OperationCounter | None = None,
) -> tuple[int, OperationCounter]:
    """Multiply with schoolbook partial products; return product and counter."""
    ctr = counter if counter is not None else OperationCounter()

    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    if a == 0 or b == 0:
        return 0, ctr

    da = int_to_digits(a)
    db = int_to_digits(b)
    accumulator: list[int] = [0]

    for j, bj in enumerate(db):
        partial: list[int] = []
        carry = 0
        for i, ai in enumerate(da):
            product = single_digit_mult(ai, bj, ctr)
            total = product + carry
            if carry:
                ctr.add()
            partial_digit = total % 10
            carry = total // 10
            if carry:
                ctr.carry()
            partial.append(partial_digit)
        if carry:
            partial.append(carry)

        for idx, pdigit in enumerate(partial):
            add_digit_at(accumulator, j + idx, pdigit, ctr)

    return digits_to_int(accumulator), ctr

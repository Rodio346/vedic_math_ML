"""Urdhva-Tiryagbhyam column-wise multiplication (fully instrumented)."""

from __future__ import annotations

from vedic_benchmark.algorithms._digits import (
    add_digit_at,
    add_value_at_position,
    digits_to_int,
    int_to_digits,
    pad_to_length,
    single_digit_mult,
)
from vedic_benchmark.analysis.counter import OperationCounter


def multiply(
    a: int,
    b: int,
    counter: OperationCounter | None = None,
) -> tuple[int, OperationCounter]:
    """Multiply with Urdhva-Tiryagbhyam; return product and populated counter."""
    ctr = counter if counter is not None else OperationCounter()

    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    if a == 0 or b == 0:
        return 0, ctr

    da_raw = int_to_digits(a)
    db_raw = int_to_digits(b)
    n = max(len(da_raw), len(db_raw))
    da = pad_to_length(da_raw, n)
    db = pad_to_length(db_raw, n)

    result_digits: list[int] = []
    carry_digits: list[int] = []

    for k in range(2 * n - 1):
        column_digits: list[int] = []

        for i in range(n):
            j = k - i
            if 0 <= j < n:
                product = single_digit_mult(da[i], db[j], ctr)
                add_value_at_position(column_digits, 0, product, ctr)

        for idx, digit in enumerate(carry_digits):
            add_digit_at(column_digits, idx, digit, ctr)
        carry_digits = []

        if not column_digits:
            column_digits = [0]

        result_digits.append(column_digits[0])
        if len(column_digits) > 1:
            carry_digits = column_digits[1:]

    for idx, digit in enumerate(carry_digits):
        add_digit_at(result_digits, (2 * n - 1) + idx, digit, ctr)

    return digits_to_int(result_digits), ctr

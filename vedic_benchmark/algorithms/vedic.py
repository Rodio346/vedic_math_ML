"""Urdhva-Tiryagbhyam column-wise multiplication (fully instrumented).

The sutra 'vertically and crosswise' computes each output digit position k
by summing all cross-products (da[i] * db[j]) where i + j == k. Operations
within the same column are structurally independent — this is the parallelism
the sutra exposes. Carries propagate sequentially across columns.

Instrumentation contract:
  - Every single-digit multiplication calls counter.multiply() via
    single_digit_mult() in _digits.py.
  - Every single-digit addition calls counter.add() via add_digit_at() /
    add_value_at_position() in _digits.py.
  - Every carry propagation calls counter.carry() via add_digit_at() in
    _digits.py AND explicitly here when a column sum overflows into carry_digits.

This file is intentionally free of bulk Python arithmetic so the operation
counts reflect the algorithm's structure, not Python's integer machinery.
"""

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
    """Multiply a and b using Urdhva-Tiryagbhyam; return (product, counter).

    Args:
        a: Non-negative integer multiplicand.
        b: Non-negative integer multiplier.
        counter: Optional pre-existing OperationCounter. If None, a fresh one
                 is created. Passing one in allows accumulation across calls.

    Returns:
        A tuple (product, counter) where product == a * b and counter holds
        the operation counts for this multiplication.

    Raises:
        ValueError: If either operand is negative.
    """
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

    # result_digits holds the final output (LSB at index 0).
    # carry_digits holds overflow from the previous column, indexed by their
    # positional offset relative to the current column's base position.
    result_digits: list[int] = []
    carry_digits: list[int] = []

    for k in range(2 * n - 1):
        # --- Column k accumulator (LSB-first digit list) ---
        column_digits: list[int] = []

        # Collect all cross-products for column k.
        # These are structurally independent — the parallelism the sutra exposes.
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                product = single_digit_mult(da[i], db[j], ctr)
                add_value_at_position(column_digits, 0, product, ctr)

        # Fold in carries from the previous column.
        # Each carry digit is added at its relative offset within this column.
        for idx, digit in enumerate(carry_digits):
            add_digit_at(column_digits, idx, digit, ctr)

        # Reset for next column.
        carry_digits = []

        if not column_digits:
            column_digits = [0]

        # The LSB of the column sum becomes the result digit at position k.
        result_digits.append(column_digits[0])

        # Any higher digits in column_digits are carries into subsequent columns.
        # Record each carry event explicitly so the counter reflects the true
        # carry structure of the algorithm.
        if len(column_digits) > 1:
            for carry_digit in column_digits[1:]:
                if carry_digit and ctr is not None:
                    ctr.carry()
            carry_digits = column_digits[1:]

    # Final carry flush: any remaining carry digits beyond position 2N-2.
    for idx, digit in enumerate(carry_digits):
        add_digit_at(result_digits, (2 * n - 1) + idx, digit, ctr)

    return digits_to_int(result_digits), ctr
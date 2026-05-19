"""Digit-list utilities and instrumented single-digit arithmetic."""

from __future__ import annotations

from vedic_benchmark.analysis.counter import OperationCounter

_MULT_TABLE: list[list[int]] = [[i * j for j in range(10)] for i in range(10)]


def int_to_digits(value: int) -> list[int]:
    """Convert a non-negative integer to base-10 digits (LSB at index 0)."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        return [0]
    digits: list[int] = []
    while value > 0:
        digits.append(value % 10)
        value //= 10
    return digits


def pad_to_length(digits: list[int], length: int) -> list[int]:
    """Pad LSB-first digits with leading zeros (higher indices) to *length*."""
    if len(digits) >= length:
        return digits
    return digits + [0] * (length - len(digits))


def digits_to_int(digits: list[int]) -> int:
    """Convert LSB-first digit list to an integer."""
    total = 0
    place = 1
    for digit in digits:
        total += digit * place
        place *= 10
    return total


def single_digit_mult(
    d1: int,
    d2: int,
    counter: OperationCounter | None,
) -> int:
    """Multiply two decimal digits via lookup table; record one multiply op."""
    if counter is not None:
        counter.multiply()
    return _MULT_TABLE[d1][d2]


def _ensure_len(digits: list[int], length: int) -> None:
    if len(digits) < length:
        digits.extend([0] * (length - len(digits)))


def add_digit_at(
    digits: list[int],
    position: int,
    digit: int,
    counter: OperationCounter | None,
) -> list[int]:
    """Add a single digit at *position* (LSB index), propagating carries."""
    _ensure_len(digits, position + 1)
    carry = digit
    pos = position
    while carry:
        _ensure_len(digits, pos + 1)
        total = digits[pos] + carry
        if counter is not None:
            counter.add()
        digits[pos] = total % 10
        carry = total // 10
        if carry and counter is not None:
            counter.carry()
        pos += 1
    return digits


def add_value_at_position(
    digits: list[int],
    position: int,
    value: int,
    counter: OperationCounter | None,
) -> list[int]:
    """Add *value* at *position* using single-digit adds and carries only."""
    if value == 0:
        return digits
    place = position
    remaining = value
    while remaining:
        digit = remaining % 10
        remaining //= 10
        if digit:
            add_digit_at(digits, place, digit, counter)
        place += 1
    return digits

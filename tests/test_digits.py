"""Digit conversion and padding utilities."""

from __future__ import annotations

from vedic_benchmark.algorithms._digits import (
    digits_to_int,
    int_to_digits,
    pad_to_length,
)


def test_int_to_digits_zero() -> None:
    assert int_to_digits(0) == [0]


def test_int_to_digits_lsb_first() -> None:
    assert int_to_digits(123) == [3, 2, 1]


def test_digits_to_int_round_trip() -> None:
    for value in (0, 1, 10, 99, 123, 999, 4712):
        assert digits_to_int(int_to_digits(value)) == value


def test_pad_to_length() -> None:
    da = int_to_digits(10)
    db = int_to_digits(999)
    n = max(len(da), len(db))
    assert n == 3
    assert pad_to_length(da, n) == [0, 1, 0]
    assert pad_to_length(db, n) == [9, 9, 9]

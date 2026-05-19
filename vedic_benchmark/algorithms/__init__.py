"""Multiplication algorithm implementations."""

from vedic_benchmark.algorithms.native import multiply as native_multiply
from vedic_benchmark.algorithms.schoolbook import multiply as schoolbook_multiply
from vedic_benchmark.algorithms.vedic import multiply as vedic_multiply

__all__ = [
    "native_multiply",
    "schoolbook_multiply",
    "vedic_multiply",
]

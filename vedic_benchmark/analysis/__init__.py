"""Analysis utilities: operation counting and theoretical depth metrics."""

from vedic_benchmark.analysis.counter import OperationCounter
from vedic_benchmark.analysis.depth import (
    DepthMetrics,
    compare_depth,
    schoolbook_depth_metrics,
    vedic_depth_metrics,
)

__all__ = [
    "OperationCounter",
    "DepthMetrics",
    "compare_depth",
    "schoolbook_depth_metrics",
    "vedic_depth_metrics",
]

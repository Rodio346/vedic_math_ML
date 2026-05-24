"""Structural comparison of attention tiling DAGs."""

from attention_dag.analysis.compare import (
    CompareResult,
    TileUnitMetrics,
    compare_strategies,
    compare_strategies_analytical,
    tile_unit_metrics,
)

__all__ = [
    "CompareResult",
    "TileUnitMetrics",
    "compare_strategies",
    "compare_strategies_analytical",
    "tile_unit_metrics",
]

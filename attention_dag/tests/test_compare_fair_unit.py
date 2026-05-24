"""Fair tile-unit comparison metrics (not global DAG CP)."""

from __future__ import annotations

import math

import pytest

from attention_dag.analysis.compare import (
    FLASH_INTER_TILE,
    VEDIC_INTER_TILE,
    compare_strategies,
    compare_strategies_analytical,
    tile_unit_metrics,
)


def test_tile_unit_cp_equal_at_d_64() -> None:
    m = tile_unit_metrics(64, 8)
    assert m.flash_per_tile_cp == 64
    assert m.vedic_total_dim_cp == 64
    assert m.vedic_per_embed_tile_cp == 8


def test_concurrent_tiles_d_64_tile_d_8() -> None:
    m = tile_unit_metrics(64, 8)
    assert m.concurrent_tiles_flash == 1
    assert m.concurrent_tiles_vedic == 8


def test_inter_tile_dependency_labels() -> None:
    m = tile_unit_metrics(4, 2)
    assert m.inter_tile_dependency_flash == FLASH_INTER_TILE
    assert m.inter_tile_dependency_vedic == VEDIC_INTER_TILE


@pytest.mark.parametrize(
    "tile_d,expected_concurrent,expected_verdict",
    [
        (8, 8, "vedic_shorter"),
        (16, 4, "vedic_shorter"),
        (32, 2, "vedic_shorter"),
        (64, 1, "identical"),
    ],
)
def test_structural_verdict_from_concurrent_tiles(
    tile_d: int,
    expected_concurrent: int,
    expected_verdict: str,
) -> None:
    r = compare_strategies_analytical(128, 64, 16, 16, tile_d)
    assert r.concurrent_tiles_vedic == expected_concurrent
    assert r.structural_verdict == expected_verdict
    assert r.efficiency_ratio == float(expected_concurrent)
    assert r.flash_critical_path == r.vedic_critical_path == 64


def test_analytical_matches_small_dag_compare() -> None:
    """Tile-unit fields match between analytical and built DAG paths."""
    a = compare_strategies_analytical(4, 4, 1, 2, 2)
    b = compare_strategies(4, 4, 1, 2, 2, [1, math.inf])
    assert a.flash_critical_path == b.flash_critical_path == 4
    assert a.concurrent_tiles_vedic == b.concurrent_tiles_vedic == 2
    assert a.structural_verdict == b.structural_verdict == "vedic_shorter"
    assert a.dag_global_flash_cp is not None
    assert b.dag_global_flash_cp is not None
    assert b.dag_global_flash_cp == 7

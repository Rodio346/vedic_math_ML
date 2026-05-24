"""Structural comparison and crossover analysis for attention DAGs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from attention_dag.dag._build import count_by_type, topological_sort
from attention_dag.dag.flashattn_dag import build_flash_dag
from attention_dag.dag.node import Node, NodeType
from attention_dag.dag.scheduler import ScheduleResult, critical_path_length, schedule
from attention_dag.dag.vedic_attn_dag import build_vedic_attn_dag

FLASH_INTER_TILE = "sequential"
VEDIC_INTER_TILE = "none"


@dataclass
class TileUnitMetrics:
    """Fair comparison unit: one tile of score computations (CP along D)."""

    per_tile_cp: int
    flash_per_tile_cp: int
    vedic_per_embed_tile_cp: int
    vedic_total_dim_cp: int
    concurrent_tiles_flash: int
    concurrent_tiles_vedic: int
    inter_tile_dependency_flash: str
    inter_tile_dependency_vedic: str


@dataclass
class CompareResult:
    flash_critical_path: int
    vedic_critical_path: int
    flash_per_tile_cp: int
    vedic_per_embed_tile_cp: int
    concurrent_tiles_flash: int
    concurrent_tiles_vedic: int
    inter_tile_dependency_flash: str
    inter_tile_dependency_vedic: str
    flash_parallel_width: int
    vedic_parallel_width: int
    flash_total_ops: int
    vedic_total_ops: int
    efficiency_ratio: float
    crossover_workers: int | None
    speedup_curve_flash: list[tuple[int, int]]
    speedup_curve_vedic: list[tuple[int, int]]
    structural_verdict: str
    dag_global_flash_cp: int | None = None
    dag_global_vedic_cp: int | None = None


def tile_unit_metrics(d_head: int, tile_d: int) -> TileUnitMetrics:
    """
    Fair CP unit: one Br×Bc score tile (Flash) vs one embed-tile wave (Vedic).

    Both perform a full D-dimensional dot-product reduction per score; per-tile
    depth along D is D steps (1 MULT wave + D-1 ADDs). Vedic exposes D/tile_d
    concurrent embed-tile rounds with no cross-tile seq dependency; Flash key
  tiles along seq_len are sequential (online softmax).
    """
    if d_head < 1 or tile_d < 1:
        raise ValueError("d_head and tile_d must be >= 1")

    per_tile_cp = d_head
    vedic_embed_rounds = (d_head + tile_d - 1) // tile_d

    return TileUnitMetrics(
        per_tile_cp=per_tile_cp,
        flash_per_tile_cp=per_tile_cp,
        vedic_per_embed_tile_cp=tile_d,
        vedic_total_dim_cp=per_tile_cp,
        concurrent_tiles_flash=1,
        concurrent_tiles_vedic=vedic_embed_rounds,
        inter_tile_dependency_flash=FLASH_INTER_TILE,
        inter_tile_dependency_vedic=VEDIC_INTER_TILE,
    )


def count_type(nodes: dict[int, Node], node_type: NodeType) -> int:
    return count_by_type(nodes, node_type)


def longest_path(nodes: dict[int, Node], topo_order: list[int]) -> int:
    del topo_order
    return critical_path_length(nodes)


def assert_no_cycles(nodes: dict[int, Node], topo_order: list[int]) -> None:
    if len(topo_order) != len(nodes):
        raise AssertionError("topological order length mismatch")
    topological_sort(nodes)


def max_parallel_width(nodes: dict[int, Node], topo_order: list[int]) -> int:
    """Maximum number of ready nodes at any single scheduling step."""
    if not nodes:
        return 0

    completed: set[int] = set()
    remaining = set(nodes.keys())
    peak = 0

    while remaining:
        ready = [
            nid
            for nid in topo_order
            if nid in remaining and all(d in completed for d in nodes[nid].deps)
        ]
        if not ready:
            raise ValueError("deadlock while measuring parallel width")
        peak = max(peak, len(ready))
        for nid in ready:
            completed.add(nid)
            remaining.remove(nid)

    return peak


def _dedupe_worker_counts(worker_counts: list[int | float]) -> list[int | float]:
    seen: set[int | float] = set()
    unique: list[int | float] = []
    for w in worker_counts:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def _finite_curve(
    flash_sched: dict[int | float, ScheduleResult],
    vedic_sched: dict[int | float, ScheduleResult],
    algorithm: str,
) -> list[tuple[int, int]]:
    sched = flash_sched if algorithm == "flash" else vedic_sched
    curve: list[tuple[int, int]] = []
    for workers, result in sorted(
        sched.items(),
        key=lambda item: (item[0] == math.inf, item[0]),
    ):
        if workers != math.inf:
            curve.append((int(workers), result.completion_time))
    return curve


def _find_crossover(
    vedic_curve: list[tuple[int, int]],
    flash_curve: list[tuple[int, int]],
) -> int | None:
    flash_by_w = dict(flash_curve)
    for workers, vedic_time in sorted(vedic_curve, key=lambda x: x[0]):
        if workers == 1:
            continue
        flash_time = flash_by_w.get(workers)
        if flash_time is not None and vedic_time < flash_time:
            return workers
    return None


def _structural_verdict_from_concurrent(
    concurrent_tiles_flash: int,
    concurrent_tiles_vedic: int,
) -> str:
    """Verdict uses concurrent tile rounds, not mismatched global DAG CP."""
    if concurrent_tiles_vedic > concurrent_tiles_flash:
        return "vedic_shorter"
    if concurrent_tiles_vedic < concurrent_tiles_flash:
        return "flash_shorter"
    return "identical"


def _efficiency_ratio_concurrent(
    concurrent_tiles_flash: int,
    concurrent_tiles_vedic: int,
) -> float:
    if concurrent_tiles_flash == 0:
        return 0.0
    return concurrent_tiles_vedic / concurrent_tiles_flash


def analytical_total_ops(seq_len: int, d_head: int) -> int:
    return seq_len * seq_len * (2 * d_head - 1)


def analytical_parallel_width(
    seq_len: int,
    d_head: int,
    tile_d: int,
    tile_r: int,
    tile_c: int,
) -> tuple[int, int]:
    """Upper-bound peak ready nodes (all MULTs fire in one wave)."""
    flash_w = min(seq_len * seq_len * d_head, tile_r * tile_c * d_head)
    vedic_w = seq_len * seq_len * min(tile_d, d_head)
    return flash_w, vedic_w


def dag_global_flash_critical_path(
    seq_len: int,
    d_head: int,
    tile_c: int,
) -> int:
    """Full-graph CP per query row (seq-tile softmax chain); not the fair tile unit."""
    key_tiles = max(1, (seq_len + tile_c - 1) // tile_c)
    if key_tiles <= 1:
        return d_head
    return 2 * (d_head - 1) + (key_tiles - 1)


def compare_strategies_analytical(
    seq_len: int,
    d_head: int,
    tile_r: int,
    tile_c: int,
    tile_d: int,
) -> CompareResult:
    """Fast comparison using tile-unit metrics (no DAG build)."""
    units = tile_unit_metrics(d_head, tile_d)
    total = analytical_total_ops(seq_len, d_head)
    flash_w, vedic_w = analytical_parallel_width(
        seq_len, d_head, tile_d, tile_r, tile_c
    )
    eff = _efficiency_ratio_concurrent(
        units.concurrent_tiles_flash,
        units.concurrent_tiles_vedic,
    )
    return CompareResult(
        flash_critical_path=units.flash_per_tile_cp,
        vedic_critical_path=units.vedic_total_dim_cp,
        flash_per_tile_cp=units.flash_per_tile_cp,
        vedic_per_embed_tile_cp=units.vedic_per_embed_tile_cp,
        concurrent_tiles_flash=units.concurrent_tiles_flash,
        concurrent_tiles_vedic=units.concurrent_tiles_vedic,
        inter_tile_dependency_flash=units.inter_tile_dependency_flash,
        inter_tile_dependency_vedic=units.inter_tile_dependency_vedic,
        flash_parallel_width=flash_w,
        vedic_parallel_width=vedic_w,
        flash_total_ops=total,
        vedic_total_ops=total,
        efficiency_ratio=eff,
        crossover_workers=None,
        speedup_curve_flash=[],
        speedup_curve_vedic=[],
        structural_verdict=_structural_verdict_from_concurrent(
            units.concurrent_tiles_flash,
            units.concurrent_tiles_vedic,
        ),
        dag_global_flash_cp=dag_global_flash_critical_path(seq_len, d_head, tile_c),
        dag_global_vedic_cp=None,
    )


def compare_strategies(
    seq_len: int,
    d_head: int,
    tile_r: int,
    tile_c: int,
    tile_d: int,
    worker_counts: list[int | float],
) -> CompareResult:
    """Build both DAGs, run scheduler, return comparison with fair tile-unit metrics."""
    flash_nodes, flash_topo = build_flash_dag(seq_len, d_head, tile_r, tile_c)
    vedic_nodes, vedic_topo = build_vedic_attn_dag(seq_len, d_head, tile_d)

    flash_total = len(flash_nodes)
    vedic_total = len(vedic_nodes)
    if flash_total != vedic_total:
        raise ValueError(
            f"total ops mismatch: flash={flash_total} vedic={vedic_total}"
        )

    units = tile_unit_metrics(d_head, tile_d)
    counts = _dedupe_worker_counts(worker_counts)
    flash_sched = {w: schedule(flash_nodes, flash_topo, w) for w in counts}
    vedic_sched = {w: schedule(vedic_nodes, vedic_topo, w) for w in counts}

    flash_curve = _finite_curve(flash_sched, vedic_sched, "flash")
    vedic_curve = _finite_curve(flash_sched, vedic_sched, "vedic")

    return CompareResult(
        flash_critical_path=units.flash_per_tile_cp,
        vedic_critical_path=units.vedic_total_dim_cp,
        flash_per_tile_cp=units.flash_per_tile_cp,
        vedic_per_embed_tile_cp=units.vedic_per_embed_tile_cp,
        concurrent_tiles_flash=units.concurrent_tiles_flash,
        concurrent_tiles_vedic=units.concurrent_tiles_vedic,
        inter_tile_dependency_flash=units.inter_tile_dependency_flash,
        inter_tile_dependency_vedic=units.inter_tile_dependency_vedic,
        flash_parallel_width=max_parallel_width(flash_nodes, flash_topo),
        vedic_parallel_width=max_parallel_width(vedic_nodes, vedic_topo),
        flash_total_ops=flash_total,
        vedic_total_ops=vedic_total,
        efficiency_ratio=_efficiency_ratio_concurrent(
            units.concurrent_tiles_flash,
            units.concurrent_tiles_vedic,
        ),
        crossover_workers=_find_crossover(vedic_curve, flash_curve),
        speedup_curve_flash=flash_curve,
        speedup_curve_vedic=vedic_curve,
        structural_verdict=_structural_verdict_from_concurrent(
            units.concurrent_tiles_flash,
            units.concurrent_tiles_vedic,
        ),
        dag_global_flash_cp=flash_sched[math.inf].completion_time,
        dag_global_vedic_cp=vedic_sched[math.inf].completion_time,
    )

"""Discrete-time parallel worker scheduler for operation DAGs."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from phase2b.dag.node import Node


@dataclass
class ScheduleResult:
    completion_time: int
    utilisation: float
    ops_per_step: list[int]


def critical_path_length(nodes: dict[int, Node]) -> int:
    """Longest path length (each node duration=1) with unlimited parallelism."""
    if not nodes:
        return 0

    dist: dict[int, int] = {nid: 1 for nid in nodes}
    order = _reverse_topological_order(nodes)

    for nid in order:
        node = nodes[nid]
        if node.deps:
            dist[nid] = 1 + max(dist[d] for d in node.deps)

    return max(dist.values())


def _reverse_topological_order(nodes: dict[int, Node]) -> list[int]:
    """Topological order with dependents before dependents' sources (for DP)."""
    in_degree = {nid: len(nodes[nid].deps) for nid in nodes}
    queue: deque[int] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[int] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for other_id, other in nodes.items():
            if nid in other.deps:
                in_degree[other_id] -= 1
                if in_degree[other_id] == 0:
                    queue.append(other_id)

    if len(order) != len(nodes):
        raise ValueError("cycle detected in DAG")
    return order


def schedule(
    nodes: dict[int, Node],
    topo_order: list[int],
    workers: int | float,
) -> ScheduleResult:
    """Simulate DAG execution with a fixed worker pool."""
    if not nodes:
        return ScheduleResult(0, 0.0, [])

    if workers == math.inf:
        cp = critical_path_length(nodes)
        total_ops = len(nodes)
        util = total_ops / (math.inf * cp) if cp else 0.0
        return ScheduleResult(cp, util, [])

    w = int(workers)
    if w < 1:
        raise ValueError("workers must be >= 1 or inf")

    completed: set[int] = set()
    remaining = set(nodes.keys())
    ops_per_step: list[int] = []
    time_steps = 0

    while remaining:
        ready = [
            nid
            for nid in topo_order
            if nid in remaining and all(d in completed for d in nodes[nid].deps)
        ]
        if not ready:
            raise ValueError("deadlock: no ready nodes but work remains")

        batch = ready[:w]
        for nid in batch:
            completed.add(nid)
            remaining.remove(nid)

        ops_per_step.append(len(batch))
        time_steps += 1

    total_ops = len(nodes)
    utilisation = total_ops / (w * time_steps) if time_steps else 0.0
    return ScheduleResult(time_steps, utilisation, ops_per_step)


def worker_counts(n_digits: int) -> list[int | float]:
    """Worker sweep for schedule_all_workers."""
    n = max(n_digits, 1)
    counts: list[int | float] = [1, 2, 4, 8, n, n // 2 + 1, 2 * n, n * n, math.inf]
    if n >= 7:
        counts.extend([n + 1, n + 2, n + 4])
    seen: set[int | float] = set()
    unique: list[int | float] = []
    for c in counts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def schedule_all_workers(
    nodes: dict[int, Node],
    topo_order: list[int],
    n_digits: int,
) -> dict[int | float, ScheduleResult]:
    """Run schedule for all standard worker counts."""
    return {
        w: schedule(nodes, topo_order, w)
        for w in worker_counts(n_digits)
    }

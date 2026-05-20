"""Crossover analysis between Vedic and schoolbook DAG schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

from phase2b.dag.scheduler import ScheduleResult


@dataclass
class CrossoverResult:
    crossover_workers: int | None
    vedic_min_completion: int
    school_min_completion: int
    efficiency_ratio: float
    speedup_curve_vedic: list[tuple[int, int]]
    speedup_curve_school: list[tuple[int, int]]
    note: str = ""


def _finite_curve(
    schedule: dict[int | float, ScheduleResult],
) -> list[tuple[int, int]]:
    curve: list[tuple[int, int]] = []
    for workers, result in sorted(
        schedule.items(),
        key=lambda item: (item[0] == math.inf, item[0]),
    ):
        if workers != math.inf:
            curve.append((int(workers), result.completion_time))
    return curve


def find_crossover(
    vedic_schedule: dict[int | float, ScheduleResult],
    schoolbook_schedule: dict[int | float, ScheduleResult],
) -> CrossoverResult:
    """Find minimum worker count where Vedic completion < schoolbook (W > 1)."""
    vedic_min = vedic_schedule[math.inf].completion_time
    school_min = schoolbook_schedule[math.inf].completion_time
    efficiency_ratio = school_min / vedic_min if vedic_min else 0.0

    vedic_curve = _finite_curve(vedic_schedule)
    school_curve = _finite_curve(schoolbook_schedule)
    school_by_w = dict(school_curve)

    crossover: int | None = None
    for workers, vedic_time in sorted(vedic_curve, key=lambda x: x[0]):
        if workers == 1:
            continue
        school_time = school_by_w.get(workers)
        if school_time is not None and vedic_time < school_time:
            crossover = workers
            break

    note = ""
    if crossover is None:
        parallel_points = [
            (w, v, school_by_w[w])
            for w, v in vedic_curve
            if w != 1 and w in school_by_w
        ]
        if parallel_points and all(v < s for _, v, s in parallel_points):
            note = "Outcome C: Vedic < schoolbook at all tested worker counts (W > 1)."
        elif parallel_points and all(v >= s for _, v, s in parallel_points):
            note = (
                "Outcome B: No crossover in tested range; schoolbook faster "
                "or tied at all tested worker counts (W > 1)."
            )
        else:
            note = "No crossover in tested range; compare speedup curves."

    return CrossoverResult(
        crossover_workers=crossover,
        vedic_min_completion=vedic_min,
        school_min_completion=school_min,
        efficiency_ratio=efficiency_ratio,
        speedup_curve_vedic=vedic_curve,
        speedup_curve_school=school_curve,
        note=note,
    )

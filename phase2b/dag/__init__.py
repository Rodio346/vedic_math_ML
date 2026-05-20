"""DAG construction and parallel scheduling for multiplication algorithms."""

from phase2b.dag.node import Node, NodeType
from phase2b.dag.scheduler import (
    ScheduleResult,
    critical_path_length,
    schedule,
    schedule_all_workers,
)
from phase2b.dag.schoolbook_dag import build_schoolbook_dag
from phase2b.dag.vedic_dag import build_vedic_dag

__all__ = [
    "Node",
    "NodeType",
    "ScheduleResult",
    "build_schoolbook_dag",
    "build_vedic_dag",
    "critical_path_length",
    "schedule",
    "schedule_all_workers",
]

"""DAG node types for single-digit operation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class NodeType(Enum):
    MULT = auto()
    ADD = auto()
    CARRY = auto()


@dataclass
class Node:
    node_id: int
    node_type: NodeType
    deps: list[int] = field(default_factory=list)
    duration: int = 1
    label: str = ""

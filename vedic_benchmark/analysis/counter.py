"""Operation counter for instrumented multiplication algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperationCounter:
    """Tracks single-digit arithmetic operations during multiplication."""

    multiplications: int = 0
    additions: int = 0
    carry_propagations: int = 0

    @property
    def total_ops(self) -> int:
        """Sum of all counted operation types."""
        return self.multiplications + self.additions + self.carry_propagations

    def multiply(self) -> None:
        """Record one single-digit multiplication."""
        self.multiplications += 1

    def add(self, count: int = 1) -> None:
        """Record one or more single-digit additions."""
        self.additions += count

    def carry(self, count: int = 1) -> None:
        """Record one or more carry-propagation steps."""
        self.carry_propagations += count

    def reset(self) -> None:
        """Clear all counters."""
        self.multiplications = 0
        self.additions = 0
        self.carry_propagations = 0

    def summary(self) -> str:
        """Human-readable summary of operation counts."""
        return (
            f"multiplications={self.multiplications}, "
            f"additions={self.additions}, "
            f"carry_propagations={self.carry_propagations}, "
            f"total_ops={self.total_ops}"
        )

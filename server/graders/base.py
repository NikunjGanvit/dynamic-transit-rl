"""
Base Grader — Abstract base class for task graders.

Each grader evaluates agent performance on a task and produces
a deterministic score in [0.0, 1.0].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..simulation.engine import SimulationEngine


class BaseGrader(ABC):
    """Abstract base class for all task graders."""

    @abstractmethod
    def grade(self, engine: SimulationEngine) -> float:
        """
        Grade the agent's performance.
        
        Args:
            engine: The simulation engine after episode completion
            
        Returns:
            Score in [0.0, 1.0]
        """
        ...

    def _clamp(self, value: float) -> float:
        """Clamp score to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

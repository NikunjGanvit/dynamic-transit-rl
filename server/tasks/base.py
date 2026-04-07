"""
Base Task — Abstract base class for task definitions.

Each task provides a configuration (initial state, events, constraints)
and an objective description for the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTask(ABC):
    """Abstract base class for all transit environment tasks."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Unique identifier for this task."""
        ...

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """Difficulty level: easy, medium, hard, expert."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the task objective."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get task configuration for the simulation engine.
        
        Returns:
            Dict with keys: seed, max_ticks, events, initial_queues,
            fleet_overrides, reward_weights
        """
        ...

    @abstractmethod
    def get_objective_prompt(self) -> str:
        """Get the objective prompt shown to the agent."""
        ...

    def to_dict(self) -> dict:
        """Serialize task info."""
        return {
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "description": self.description,
            "objective": self.get_objective_prompt(),
        }

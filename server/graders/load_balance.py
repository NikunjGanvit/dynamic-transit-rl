"""
Load Balance Grader — Evaluates Task 2 (Medium).

Scores based on how well the agent balanced load across the system,
especially during the rush hour event.
"""

from __future__ import annotations

import math

from ..simulation.engine import SimulationEngine
from .base import BaseGrader


class LoadBalanceGrader(BaseGrader):
    """
    Grader for the Balance System Load task.
    
    Score composition:
    - 40%: Variance reduction (lower variance = better)
    - 25%: Throughput efficiency
    - 20%: Rush hour handling (queue sizes during event)
    - 15%: Overall satisfaction
    """

    def grade(self, engine: SimulationEngine) -> float:
        initial_metrics = engine.get_initial_metrics()
        final_metrics = engine.get_current_metrics()

        if initial_metrics is None:
            return 0.0

        # --- Variance Score (40%) ---
        # Lower final variance = better
        max_acceptable_variance = 50.0
        variance_score = max(
            0.0,
            1.0 - (final_metrics.queue_variance / max_acceptable_variance)
        )

        # --- Throughput Score (25%) ---
        total_gen = engine.passenger_gen.total_generated
        if total_gen > 0:
            throughput_score = min(
                1.0,
                engine.passenger_gen.total_boarded / total_gen
            )
        else:
            throughput_score = 1.0

        # --- Rush Hour Handling (20%) ---
        # Check if metrics improved during/after rush hour (ticks 8-14)
        metrics_history = engine._metrics_history
        if len(metrics_history) > 14:
            rush_metrics = metrics_history[8:15]
            avg_rush_queue = sum(
                m.max_queue_size for m in rush_metrics
            ) / len(rush_metrics)
            # Good if max queue during rush < 20
            rush_score = max(0.0, 1.0 - (avg_rush_queue / 30.0))
        else:
            rush_score = 0.5  # neutral if not enough data

        # --- Satisfaction Score (15%) ---
        satisfaction_score = final_metrics.avg_satisfaction

        # --- Compose ---
        score = (
            0.40 * variance_score
            + 0.25 * throughput_score
            + 0.20 * rush_score
            + 0.15 * satisfaction_score
        )

        return self._clamp(score)

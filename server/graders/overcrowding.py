"""
Overcrowding Grader — Evaluates Task 1 (Easy).

Multi-signal grading based on throughput, queue management, wait time,
and satisfaction relative to a no-action baseline.
"""

from __future__ import annotations

from ..simulation.engine import SimulationEngine
from .base import BaseGrader


class OvercrowdingGrader(BaseGrader):
    """
    Grader for the Reduce Overcrowding task.
    
    Score composition:
    - 30%: Throughput (passengers served / generated)
    - 25%: Queue control (inverse of average queue across all stops)
    - 20%: Max queue reduction (lower final max queue = better)
    - 15%: Satisfaction score
    - 10%: Abandonment prevention
    """

    def grade(self, engine: SimulationEngine) -> float:
        final_metrics = engine.get_current_metrics()

        # --- Throughput Score (30%) ---
        total_gen = engine.passenger_gen.total_generated
        if total_gen > 0:
            throughput_score = min(1.0, engine.passenger_gen.total_boarded / total_gen)
        else:
            throughput_score = 1.0

        # --- Queue Control Score (25%) ---
        # Average queue across all stops, lower is better
        queue_sizes = []
        for sid in engine.network.stops:
            queue_sizes.append(engine.passenger_gen.get_queue_size(sid))
        avg_queue = sum(queue_sizes) / len(queue_sizes) if queue_sizes else 0
        # Normalize: 0 avg queue = 1.0, 15+ avg queue = 0.0
        queue_score = max(0.0, 1.0 - (avg_queue / 15.0))

        # --- Max Queue Score (20%) ---
        # Lower final max queue = better
        max_queue = final_metrics.max_queue_size
        # Normalize: 0 max queue = 1.0, 30+ max queue = 0.0
        max_queue_score = max(0.0, 1.0 - (max_queue / 30.0))

        # --- Satisfaction Score (15%) ---
        satisfaction_score = final_metrics.avg_satisfaction

        # --- Abandonment Prevention (10%) ---
        if total_gen > 0:
            abandon_ratio = engine.passenger_gen.total_abandoned / total_gen
            abandon_score = max(0.0, 1.0 - (abandon_ratio * 3))
        else:
            abandon_score = 1.0

        # --- Compose ---
        score = (
            0.30 * throughput_score
            + 0.25 * queue_score
            + 0.20 * max_queue_score
            + 0.15 * satisfaction_score
            + 0.10 * abandon_score
        )

        return self._clamp(score)

"""
Demand Spike Grader — Evaluates Task 3 (Hard).

Tests the agent's ability to DETECT and RESPOND to hidden events.
Specifically measures:
- Detection speed (did the agent take action after events started?)
- Recovery quality (how well the system recovered after disruptions)
- Service under pressure (throughput during crisis periods)
- Abandonment prevention
"""

from __future__ import annotations

from ..simulation.engine import SimulationEngine, BusStatus
from .base import BaseGrader


class DemandSpikeGrader(BaseGrader):
    """
    Grader for the Handle Dynamic Disruptions task.
    
    Score composition:
    - 25%: Detection & response speed
    - 25%: Service during crisis (passengers served during events)
    - 25%: Queue recovery (final state quality)
    - 15%: Abandonment prevention
    - 10%: Resource adaptation (did agent replace broken bus?)
    """

    def grade(self, engine: SimulationEngine) -> float:
        final_metrics = engine.get_current_metrics()
        metrics_history = engine._metrics_history
        action_log = engine._action_log

        # --- Detection & Response Speed (25%) ---
        # Concert starts at tick 4. Did the agent take any action between ticks 4-7?
        # Weather at tick 8. Breakdown at tick 10.
        early_actions = [a for a in action_log if 4 <= a.get("tick", 0) <= 7]
        mid_actions = [a for a in action_log if 8 <= a.get("tick", 0) <= 12]
        
        # Score based on how many response windows the agent used
        response_windows_used = 0
        if len(early_actions) >= 1:  # Reacted to concert
            response_windows_used += 1
        if len(early_actions) >= 2:  # Strong reaction
            response_windows_used += 1
        if len(mid_actions) >= 1:  # Reacted to weather/breakdown
            response_windows_used += 1
        if len(mid_actions) >= 2:  # Strong reaction
            response_windows_used += 1
        
        response_score = min(1.0, response_windows_used / 4.0)

        # --- Service During Crisis (25%) ---
        # How many passengers were served during the crisis period (ticks 4-18)?
        total_gen = engine.passenger_gen.total_generated
        if total_gen > 0:
            service_score = min(1.0, engine.passenger_gen.total_boarded / total_gen)
        else:
            service_score = 1.0

        # --- Queue Recovery (25%) ---
        # Final system state quality
        final_max = final_metrics.max_queue_size
        final_avg_wait = final_metrics.avg_wait_time
        
        # Lower max queue = better (threshold: 20)
        queue_score = max(0.0, 1.0 - (final_max / 25.0))
        # Lower wait time = better (threshold: 8 ticks)
        wait_score = max(0.0, 1.0 - (final_avg_wait / 8.0))
        recovery_score = 0.5 * queue_score + 0.5 * wait_score

        # --- Abandonment Prevention (15%) ---
        if total_gen > 0:
            abandon_ratio = engine.passenger_gen.total_abandoned / total_gen
            abandon_score = max(0.0, 1.0 - (abandon_ratio * 2.5))
        else:
            abandon_score = 1.0

        # --- Resource Adaptation (10%) ---
        # Did the agent dispatch a replacement after B01 broke down at tick 10?
        post_breakdown_dispatches = [
            a for a in action_log
            if a.get("tick", 0) >= 10 and a.get("action") in ("dispatch_bus", "reassign_bus")
        ]
        adaptation_score = min(1.0, len(post_breakdown_dispatches) / 2.0)

        # --- Compose ---
        score = (
            0.25 * response_score
            + 0.25 * service_score
            + 0.25 * recovery_score
            + 0.15 * abandon_score
            + 0.10 * adaptation_score
        )

        return self._clamp(score)

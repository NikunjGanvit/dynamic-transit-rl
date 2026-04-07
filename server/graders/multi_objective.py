"""
Multi-Objective Grader — Evaluates Task 4 (Expert).

Tests the agent's ability to survive cascading crises with zero
advance information. Measures across 5 dimensions with strict thresholds.
"""

from __future__ import annotations

from ..simulation.engine import SimulationEngine, BusStatus
from .base import BaseGrader


class MultiObjectiveGrader(BaseGrader):
    """
    Grader for the Multi-Crisis Management task.
    
    Score composition (strict Pareto):
    - 20%: Throughput under cascading pressure
    - 20%: Wait time management
    - 20%: Route coverage maintenance (no route abandoned)
    - 20%: Crisis response quality (actions taken in response windows)
    - 10%: Passenger satisfaction
    - 10%: Fleet management (broke bus replaced? reserves maintained?)
    
    Penalties:
    - Any route with ZERO active buses: -0.15 per route
    - High abandonment: scales with ratio
    """

    def grade(self, engine: SimulationEngine) -> float:
        final_metrics = engine.get_current_metrics()
        action_log = engine._action_log

        # --- Throughput (20%) ---
        total_gen = engine.passenger_gen.total_generated
        if total_gen > 0:
            throughput_score = min(1.0, engine.passenger_gen.total_boarded / total_gen)
        else:
            throughput_score = 1.0

        # --- Wait Time (20%) ---
        # Stricter threshold for expert: 6 ticks max acceptable
        wait_score = max(0.0, 1.0 - (final_metrics.avg_wait_time / 6.0))

        # --- Route Coverage (20%) ---
        routes_served = set()
        for bus in engine.buses.values():
            if bus.status == BusStatus.ACTIVE and bus.route_id:
                routes_served.add(bus.route_id)
        total_routes = len(engine.network.routes)
        
        if total_routes > 0:
            coverage_ratio = len(routes_served) / total_routes
        else:
            coverage_ratio = 0.0
        
        # Harsh penalty for missing routes
        unserved_routes = total_routes - len(routes_served)
        coverage_score = max(0.0, coverage_ratio - (unserved_routes * 0.15))

        # --- Crisis Response Quality (20%) ---
        # 6 events happen at ticks: 2, 4, 6, 9, 12, 14
        # Did the agent take responsive actions within 2 ticks?
        crisis_windows = [
            (2, 4),   # Construction
            (4, 6),   # B03 breakdown
            (6, 8),   # Rush hour
            (9, 11),  # Storm
            (12, 14), # B06 breakdown
            (14, 16), # Concert
        ]
        
        windows_responded = 0
        for start, end in crisis_windows:
            window_actions = [
                a for a in action_log
                if start <= a.get("tick", 0) <= end
                and a.get("action") != "skip"
            ]
            if len(window_actions) >= 1:
                windows_responded += 1
        
        crisis_score = windows_responded / len(crisis_windows)

        # --- Satisfaction (10%) ---
        satisfaction_score = final_metrics.avg_satisfaction

        # --- Fleet Management (10%) ---
        # Did agent dispatch replacements for broken buses?
        post_break_actions = [
            a for a in action_log
            if a.get("tick", 0) >= 4
            and a.get("action") in ("dispatch_bus", "reassign_bus")
        ]
        fleet_score = min(1.0, len(post_break_actions) / 4.0)

        # --- Penalties ---
        # High abandonment
        if total_gen > 0:
            abandon_ratio = engine.passenger_gen.total_abandoned / total_gen
            abandon_penalty = min(0.20, abandon_ratio * 0.5)
        else:
            abandon_penalty = 0.0

        # --- Compose ---
        score = (
            0.20 * throughput_score
            + 0.20 * wait_score
            + 0.20 * coverage_score
            + 0.20 * crisis_score
            + 0.10 * satisfaction_score
            + 0.10 * fleet_score
            - abandon_penalty
        )

        return self._clamp(score)

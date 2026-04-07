"""
Task 2 — Balance System Load (Medium)

Scenario: Uneven bus distribution, some routes overserved.
Objective: Minimize variance in queue sizes across all stops.
Rush hour event occurs at tick 8.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseTask


class LoadBalanceTask(BaseTask):
    """Medium task: balance load across the transit system."""

    @property
    def task_name(self) -> str:
        return "balance_load"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def description(self) -> str:
        return (
            "The transit system has uneven bus distribution — Route R1 has too many buses "
            "while Route R4 is underserved. A morning rush hour event occurs at tick 8, "
            "increasing demand on Routes R1 and R2. Balance the system proactively."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "seed": 123,
            "max_ticks": 20,
            "events": [
                {
                    "type": "rush_hour",
                    "start_tick": 8,
                    "end_tick": 14,
                    "affected_stops": ["S01", "S09", "S05", "S02"],
                    "affected_routes": ["R1", "R2"],
                    "demand_multiplier": 2.5,
                    "speed_multiplier": 0.85,
                    "description": "Morning rush hour — heavy demand on central corridors",
                },
            ],
            "initial_queues": {
                "S11": 12,  # Industrial zone underserved
                "S07": 10,  # Airport road underserved
                "S01": 5,
            },
            "fleet_overrides": [
                # Overload R1 with extra bus, leave R4 short
                {"bus_id": "B07", "route_id": "R1", "position_idx": 1},
            ],
            "reward_weights": {
                "wait": 0.25,
                "occupancy": 0.25,
                "throughput": 0.20,
                "satisfaction": 0.15,
                "idle_penalty": 0.05,
                "overcrowding_penalty": 0.10,
            },
        }

    def get_objective_prompt(self) -> str:
        return (
            "TASK: Balance System Load (Medium)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SITUATION: The transit system has uneven bus distribution.\n"
            "- Route R1 (North-South Express) has 3 buses — overserved\n"
            "- Route R4 (Industrial Link) has 0 buses — underserved\n"
            "- Industrial Zone (S11) and Airport Road (S07) have growing queues\n\n"
            "WARNING: A morning rush hour event will hit at tick 8, increasing demand "
            "2.5x on central stops (S01, S09, S05, S02).\n\n"
            "OBJECTIVE: Minimize the variance in queue sizes across all stops while "
            "preparing for the rush hour surge.\n\n"
            "STRATEGY HINTS:\n"
            "- Rebalance buses from overserved to underserved routes BEFORE rush hour\n"
            "- Dispatch the depot bus to the most critical route\n"
            "- After rush hour starts, adapt by increasing frequency on affected routes\n"
            "- Monitor queue variance as your key metric"
        )

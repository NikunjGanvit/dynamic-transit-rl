"""
Task 1 — Reduce Overcrowding (Easy)

Scenario: 2 stops have high queues, other stops normal.
Objective: Reduce max queue to below threshold.
No dynamic events. Straightforward response expected.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseTask


class OvercrowdingTask(BaseTask):
    """Easy task: reduce overcrowding at congested stops."""

    @property
    def task_name(self) -> str:
        return "reduce_overcrowding"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def description(self) -> str:
        return (
            "Two stops (Central Station and Tech Park) are severely overcrowded. "
            "Dispatch additional buses or reassign existing ones to reduce queue sizes. "
            "No special events occur during this task."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "max_ticks": 20,
            "events": [],  # No dynamic events for easy task
            "initial_queues": {
                "S01": 25,  # Central Station: high queue
                "S05": 20,  # Tech Park: high queue
                "S02": 5,
                "S03": 3,
                "S09": 8,
            },
            "reward_weights": {
                "wait": 0.35,
                "occupancy": 0.20,
                "throughput": 0.25,
                "satisfaction": 0.15,
                "idle_penalty": 0.03,
                "overcrowding_penalty": 0.02,
            },
        }

    def get_objective_prompt(self) -> str:
        return (
            "TASK: Reduce Overcrowding (Easy)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SITUATION: Central Station (S01) and Tech Park (S05) have severe overcrowding "
            "with 25 and 20 passengers queued respectively.\n\n"
            "OBJECTIVE: Reduce the maximum queue size across all stops as much as possible.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- reassign_bus(bus_id, target_route_id): Move a bus to a congested route\n"
            "- dispatch_bus(route_id, bus_type): Deploy a depot bus to a route\n"
            "- increase_frequency(route_id): Speed up buses on a route\n"
            "- hold_bus(bus_id, duration): Hold bus at stop for extra boarding\n"
            "- skip_action(): Do nothing this step\n\n"
            "STRATEGY HINTS:\n"
            "- Check which routes serve the congested stops\n"
            "- Dispatch extra buses to those routes\n"
            "- Consider reassigning underutilized buses\n"
            "- Use get_system_status() to monitor progress"
        )

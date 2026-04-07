"""
Task 3 — Handle Dynamic Disruptions (Hard)

Scenario: Agent must manage the system under UNKNOWN dynamic events.
Events are NOT revealed in the prompt — the agent must detect them
by monitoring queue spikes and system status changes.

This tests:
- Situational awareness (detecting anomalies)
- Reactive decision-making (fast response to unexpected changes)
- Multi-event reasoning (overlapping disruptions)
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseTask


class DemandSpikeTask(BaseTask):
    """Hard task: handle unknown dynamic disruptions through observation."""

    @property
    def task_name(self) -> str:
        return "demand_spike"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return (
            "The transit system faces multiple dynamic disruptions that are NOT "
            "announced in advance. The agent must detect anomalies by monitoring "
            "system status and react appropriately. Events include demand surges, "
            "weather degradation, and equipment failures."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "seed": 256,
            "max_ticks": 20,
            "events": [
                # Event 1: Concert demand spike — NOT told to agent
                {
                    "type": "concert",
                    "start_tick": 4,
                    "end_tick": 14,
                    "affected_stops": ["S06", "S10", "S02"],
                    "affected_routes": ["R1"],
                    "demand_multiplier": 5.0,
                    "speed_multiplier": 1.0,
                    "description": "Major concert at Stadium — massive passenger surge",
                },
                # Event 2: Weather — NOT told to agent
                {
                    "type": "weather",
                    "start_tick": 8,
                    "end_tick": 18,
                    "affected_stops": [],
                    "affected_routes": ["R1", "R2", "R3", "R4"],
                    "demand_multiplier": 1.0,
                    "speed_multiplier": 0.6,
                    "description": "Heavy rain — all buses slowed by 40%",
                },
                # Event 3: Surprise breakdown — NOT told to agent
                {
                    "type": "breakdown",
                    "start_tick": 10,
                    "end_tick": 20,
                    "affected_stops": [],
                    "affected_routes": [],
                    "demand_multiplier": 1.0,
                    "speed_multiplier": 1.0,
                    "affected_buses": ["B01"],
                    "description": "Bus B01 engine failure mid-route",
                },
            ],
            "initial_queues": {
                "S01": 5,
                "S05": 4,
                "S09": 3,
            },
            "reward_weights": {
                "wait": 0.30,
                "occupancy": 0.20,
                "throughput": 0.25,
                "satisfaction": 0.15,
                "idle_penalty": 0.02,
                "overcrowding_penalty": 0.08,
            },
        }

    def get_objective_prompt(self) -> str:
        return (
            "TASK: Handle Dynamic Disruptions (Hard)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SITUATION: The transit system is operating normally with light queues.\n"
            "However, the city operations center has warned that MULTIPLE DISRUPTIONS\n"
            "may occur during this shift. The exact nature, timing, and severity of\n"
            "these events are UNKNOWN.\n\n"
            "⚠️  YOU WILL NOT BE TOLD WHAT EVENTS OCCUR OR WHEN.\n"
            "You must detect disruptions by monitoring system status changes:\n"
            "- Sudden queue size increases → demand surge\n"
            "- Bus speeds dropping → weather or road issues\n"
            "- Bus status changing to 'breakdown' → equipment failure\n\n"
            "OBJECTIVE: Maintain system stability by detecting and responding to\n"
            "disruptions as quickly as possible. Minimize passenger wait times\n"
            "and abandonment.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- reassign_bus(bus_id, target_route_id): Move a bus to a congested route\n"
            "- dispatch_bus(route_id, bus_type): Deploy a depot bus to a route\n"
            "- increase_frequency(route_id): Speed up buses on a route by 30%\n"
            "- hold_bus(bus_id, duration): Hold bus at stop for extra boarding\n"
            "- skip_action(): Do nothing this step\n\n"
            "CRITICAL STRATEGY:\n"
            "- Call get_system_status() EVERY step to monitor for anomalies\n"
            "- Compare queue sizes between steps to detect demand spikes\n"
            "- Check bus statuses for breakdowns\n"
            "- Watch active_events field for new disruptions\n"
            "- React FAST — delays compound quickly\n"
            "- You have limited resources — prioritize the worst-affected areas"
        )

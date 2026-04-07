"""
Task 4 — Multi-Crisis Management (Expert)

Scenario: Cascading failures with NO advance warning. Agent must handle:
- Rolling blackout affecting route operations
- Mass event causing demand explosion
- Multiple bus breakdowns
- Weather degradation
- All while maintaining coverage across ALL routes

Events are completely hidden. Agent gets ZERO hints about what will happen.
This exclusively tests strategic reasoning under deep uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseTask


class MultiObjectiveTask(BaseTask):
    """Expert task: survive cascading crises with zero advance information."""

    @property
    def task_name(self) -> str:
        return "multi_objective"

    @property
    def difficulty(self) -> str:
        return "expert"

    @property
    def description(self) -> str:
        return (
            "The city faces cascading operational crises with NO advance warning. "
            "Multiple bus breakdowns, severe weather, demand surges from a mass event, "
            "and road construction all compound. The agent must detect each crisis "
            "through observation and make rapid triage decisions with limited resources."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "seed": 512,
            "max_ticks": 20,
            "events": [
                # Phase 1 (tick 2): Construction starts — subtle slowdown
                {
                    "type": "construction",
                    "start_tick": 2,
                    "end_tick": 20,
                    "affected_stops": ["S04"],
                    "affected_routes": ["R2"],
                    "demand_multiplier": 1.5,
                    "speed_multiplier": 0.7,
                    "description": "Road construction near Hospital (S04)",
                },
                # Phase 2 (tick 4): First breakdown — loses key bus
                {
                    "type": "breakdown",
                    "start_tick": 4,
                    "end_tick": 20,
                    "affected_stops": [],
                    "affected_routes": [],
                    "demand_multiplier": 1.0,
                    "speed_multiplier": 1.0,
                    "affected_buses": ["B03"],
                    "description": "Bus B03 (articulated, 80-cap) engine failure",
                },
                # Phase 3 (tick 6): Rush hour hits — demand doubles
                {
                    "type": "rush_hour",
                    "start_tick": 6,
                    "end_tick": 14,
                    "affected_stops": ["S01", "S05", "S09", "S03"],
                    "affected_routes": ["R1", "R2"],
                    "demand_multiplier": 2.5,
                    "speed_multiplier": 0.85,
                    "description": "Morning rush — heavy demand on central corridors",
                },
                # Phase 4 (tick 9): Severe storm — ALL buses at 55% speed
                {
                    "type": "weather",
                    "start_tick": 9,
                    "end_tick": 18,
                    "affected_stops": [],
                    "affected_routes": ["R1", "R2", "R3", "R4"],
                    "demand_multiplier": 1.0,
                    "speed_multiplier": 0.55,
                    "description": "Severe thunderstorm — buses at 55% speed",
                },
                # Phase 5 (tick 12): Second breakdown — loses another bus
                {
                    "type": "breakdown",
                    "start_tick": 12,
                    "end_tick": 20,
                    "affected_stops": [],
                    "affected_routes": [],
                    "demand_multiplier": 1.0,
                    "speed_multiplier": 1.0,
                    "affected_buses": ["B06"],
                    "description": "Bus B06 brake failure during storm",
                },
                # Phase 6 (tick 14): Concert starts — 4x demand at stadium area
                {
                    "type": "concert",
                    "start_tick": 14,
                    "end_tick": 20,
                    "affected_stops": ["S06", "S10"],
                    "affected_routes": ["R1"],
                    "demand_multiplier": 4.0,
                    "speed_multiplier": 1.0,
                    "description": "Evening concert at Stadium — massive surge",
                },
            ],
            "initial_queues": {
                "S01": 6,
                "S05": 5,
                "S09": 5,
                "S03": 4,
                "S04": 3,
            },
            "reward_weights": {
                "wait": 0.25,
                "occupancy": 0.15,
                "throughput": 0.25,
                "satisfaction": 0.20,
                "idle_penalty": 0.05,
                "overcrowding_penalty": 0.10,
            },
        }

    def get_objective_prompt(self) -> str:
        return (
            "TASK: Multi-Crisis Management (Expert)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SITUATION: You are managing the city transit system during a shift\n"
            "flagged as HIGH RISK by the operations center. Intelligence suggests\n"
            "multiple cascading disruptions will occur, but specifics are CLASSIFIED.\n\n"
            "⚠️  ZERO ADVANCE INFORMATION IS PROVIDED.\n"
            "⚠️  MULTIPLE CRISES WILL OVERLAP AND COMPOUND.\n"
            "⚠️  YOU WILL LOSE CRITICAL RESOURCES MID-EPISODE.\n\n"
            "OBJECTIVES (all equally weighted):\n"
            "1. Maximize passenger throughput (passengers served)\n"
            "2. Minimize average wait time across all stops\n"
            "3. Maintain route coverage (no route left completely unserved)\n"
            "4. Maximize passenger satisfaction\n\n"
            "CONSTRAINTS:\n"
            "- Max fleet size: 12 buses\n"
            "- Broken buses CANNOT be repaired\n"
            "- Each action takes 1 tick — you only get 20 actions total\n"
            "- Events come in WAVES — prepare for escalation\n\n"
            "CRITICAL STRATEGY:\n"
            "- Monitor get_system_status() EVERY step — this is your only intelligence\n"
            "- Watch for: sudden queue spikes, bus status changes, speed drops\n"
            "- Keep reserve capacity — don't deploy everything early\n"
            "- When you lose a bus, immediately replace it\n"
            "- When demand spikes, dispatch to the affected route\n"
            "- When weather hits, increase_frequency to compensate\n"
            "- TRIAGE: you can't fix everything — focus on highest-impact actions\n"
            "- The situation WILL get worse before it gets better"
        )

"""
Events — Dynamic event system for the transit simulation.

Defines event types (rush hour, weather, concert, breakdown) that
modify simulation parameters at specific ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Types of dynamic events that affect the simulation."""
    RUSH_HOUR = "rush_hour"
    WEATHER = "weather"
    CONCERT = "concert"
    BREAKDOWN = "breakdown"
    CONSTRUCTION = "construction"


@dataclass
class Event:
    """A scheduled event that modifies simulation parameters."""
    event_type: EventType
    start_tick: int
    end_tick: int
    affected_stops: List[str]  # stop IDs affected
    affected_routes: List[str]  # route IDs affected
    demand_multiplier: float = 1.0  # multiplier on passenger demand
    speed_multiplier: float = 1.0  # multiplier on bus speed
    affected_buses: List[str] = field(default_factory=list)  # specific buses
    description: str = ""

    @property
    def is_active(self) -> bool:
        """Check if event is currently active (must be checked with tick)."""
        return True  # Checked externally

    def to_dict(self) -> dict:
        """Serialize event for observation."""
        return {
            "type": self.event_type.value,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "affected_stops": self.affected_stops,
            "affected_routes": self.affected_routes,
            "demand_multiplier": self.demand_multiplier,
            "speed_multiplier": self.speed_multiplier,
            "description": self.description,
        }


class EventManager:
    """
    Manages dynamic events in the simulation.
    
    Events are pre-scheduled based on task configuration and are
    deterministic given the same seed/task.
    """

    def __init__(self) -> None:
        self.events: List[Event] = []
        self._active_cache: List[Event] = []

    def schedule_events(self, task_events: List[Dict[str, Any]]) -> None:
        """
        Schedule events from task configuration.
        
        Args:
            task_events: List of event dicts with type, timing, and parameters
        """
        self.events = []
        for ev_config in task_events:
            event = Event(
                event_type=EventType(ev_config["type"]),
                start_tick=ev_config["start_tick"],
                end_tick=ev_config["end_tick"],
                affected_stops=ev_config.get("affected_stops", []),
                affected_routes=ev_config.get("affected_routes", []),
                demand_multiplier=ev_config.get("demand_multiplier", 1.0),
                speed_multiplier=ev_config.get("speed_multiplier", 1.0),
                affected_buses=ev_config.get("affected_buses", []),
                description=ev_config.get("description", ""),
            )
            self.events.append(event)

    def get_active_events(self, tick: int) -> List[Event]:
        """Get all events active at the given tick."""
        self._active_cache = [
            e for e in self.events
            if e.start_tick <= tick <= e.end_tick
        ]
        return self._active_cache

    def get_demand_multipliers(self, tick: int) -> Dict[str, float]:
        """
        Get per-stop demand multipliers from active events.
        Multiple events multiply together.
        """
        active = self.get_active_events(tick)
        multipliers: Dict[str, float] = {}
        for event in active:
            for stop_id in event.affected_stops:
                if stop_id in multipliers:
                    multipliers[stop_id] *= event.demand_multiplier
                else:
                    multipliers[stop_id] = event.demand_multiplier
        return multipliers

    def get_speed_multiplier(self, tick: int, route_id: str) -> float:
        """Get speed multiplier for a route from active events."""
        active = self.get_active_events(tick)
        multiplier = 1.0
        for event in active:
            if route_id in event.affected_routes or not event.affected_routes:
                if event.speed_multiplier != 1.0:
                    multiplier *= event.speed_multiplier
        return multiplier

    def get_broken_buses(self, tick: int) -> List[str]:
        """Get list of bus IDs affected by breakdown events."""
        active = self.get_active_events(tick)
        broken = []
        for event in active:
            if event.event_type == EventType.BREAKDOWN:
                broken.extend(event.affected_buses)
        return broken

    def get_active_event_descriptions(self, tick: int) -> List[dict]:
        """Get descriptions of all active events for observation."""
        active = self.get_active_events(tick)
        return [e.to_dict() for e in active]

    def get_time_multiplier(self, tick: int, total_ticks: int = 20) -> float:
        """
        Get time-of-day demand multiplier.
        Simulates morning/evening peaks in a simplified day cycle.
        
        Pattern over 20 ticks:
        - Ticks 0-3: Early morning (0.6x)
        - Ticks 4-7: Morning rush (1.5x)
        - Ticks 8-11: Midday (1.0x)
        - Ticks 12-15: Evening rush (1.4x)
        - Ticks 16-19: Night (0.7x)
        """
        if tick < 4:
            return 0.6
        elif tick < 8:
            return 1.5
        elif tick < 12:
            return 1.0
        elif tick < 16:
            return 1.4
        else:
            return 0.7

    def reset(self) -> None:
        """Clear all events."""
        self.events = []
        self._active_cache = []

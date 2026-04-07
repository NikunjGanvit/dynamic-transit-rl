"""
Bus — Vehicle model for the transit simulation.

Defines bus types (Standard, Articulated, Mini) and bus state
including position, passengers, capacity, and operational status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BusType(str, Enum):
    """Types of buses with different capacities."""
    STANDARD = "standard"       # 40 passengers
    ARTICULATED = "articulated" # 80 passengers
    MINI = "mini"               # 20 passengers


class BusStatus(str, Enum):
    """Operational status of a bus."""
    ACTIVE = "active"       # Currently operating on a route
    HELD = "held"           # Temporarily held at a stop
    DEPOT = "depot"         # Parked in depot (not in service)
    BREAKDOWN = "breakdown" # Out of service due to breakdown


# Capacity lookup
BUS_CAPACITY = {
    BusType.STANDARD: 40,
    BusType.ARTICULATED: 80,
    BusType.MINI: 20,
}


@dataclass
class Bus:
    """A bus operating in the transit system."""
    bus_id: str
    bus_type: BusType
    route_id: Optional[str]
    position_idx: int  # index into route's stop list
    passengers: int = 0
    capacity: int = 0
    status: BusStatus = BusStatus.ACTIVE
    hold_remaining: int = 0  # ticks remaining in hold
    speed_multiplier: float = 1.0  # affected by weather/traffic
    travel_progress: int = 0  # progress toward next stop (ticks elapsed)
    direction: int = 1  # 1 = forward along route, -1 = backward
    total_passengers_served: int = 0

    def __post_init__(self) -> None:
        if self.capacity == 0:
            self.capacity = BUS_CAPACITY[self.bus_type]

    @property
    def occupancy_ratio(self) -> float:
        """Current occupancy as a fraction of capacity."""
        if self.capacity == 0:
            return 0.0
        return self.passengers / self.capacity

    @property
    def available_seats(self) -> int:
        """Number of available seats."""
        return max(0, self.capacity - self.passengers)

    @property
    def is_overcrowded(self) -> bool:
        """Bus is at or over capacity."""
        return self.passengers >= self.capacity

    def board_passengers(self, count: int) -> int:
        """
        Board passengers onto the bus.
        Returns the number actually boarded (limited by capacity).
        """
        can_board = min(count, self.available_seats)
        self.passengers += can_board
        self.total_passengers_served += can_board
        return can_board

    def alight_passengers(self, count: int) -> int:
        """
        Alight passengers from the bus.
        Returns the number that actually alighted.
        """
        can_alight = min(count, self.passengers)
        self.passengers -= can_alight
        return can_alight

    def to_dict(self) -> dict:
        """Serialize bus state for observation."""
        return {
            "bus_id": self.bus_id,
            "type": self.bus_type.value,
            "route_id": self.route_id,
            "position_idx": self.position_idx,
            "passengers": self.passengers,
            "capacity": self.capacity,
            "occupancy_pct": round(self.occupancy_ratio * 100, 1),
            "status": self.status.value,
            "hold_remaining": self.hold_remaining,
            "speed_multiplier": self.speed_multiplier,
            "direction": self.direction,
            "total_served": self.total_passengers_served,
        }


def create_default_fleet() -> List[Bus]:
    """Create the default fleet of 8 buses."""
    fleet = [
        Bus(bus_id="B01", bus_type=BusType.STANDARD, route_id="R1", position_idx=0),
        Bus(bus_id="B02", bus_type=BusType.STANDARD, route_id="R1", position_idx=3),
        Bus(bus_id="B03", bus_type=BusType.ARTICULATED, route_id="R2", position_idx=0),
        Bus(bus_id="B04", bus_type=BusType.STANDARD, route_id="R2", position_idx=2),
        Bus(bus_id="B05", bus_type=BusType.STANDARD, route_id="R3", position_idx=0),
        Bus(bus_id="B06", bus_type=BusType.MINI, route_id="R3", position_idx=2),
        Bus(bus_id="B07", bus_type=BusType.STANDARD, route_id="R4", position_idx=0),
        Bus(bus_id="B08", bus_type=BusType.MINI, route_id=None, position_idx=0,
            status=BusStatus.DEPOT),
    ]
    return fleet

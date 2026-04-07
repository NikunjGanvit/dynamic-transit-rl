"""
City Network — Topology of bus stops and routes.

Defines a realistic urban transit network with 12 stops across 4 routes,
including inter-route transfer points and distance/travel-time matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Stop:
    """A bus stop in the transit network."""
    stop_id: str
    name: str
    x: float  # longitude-like coordinate
    y: float  # latitude-like coordinate
    base_demand: float  # base passengers per tick
    is_transfer: bool = False  # can transfer between routes here


@dataclass(frozen=True)
class Route:
    """A bus route connecting an ordered sequence of stops."""
    route_id: str
    name: str
    stop_ids: Tuple[str, ...]
    base_frequency: int = 3  # ticks between dispatches
    color: str = "#3b82f6"


def _distance(s1: Stop, s2: Stop) -> float:
    """Euclidean distance between two stops."""
    return math.hypot(s1.x - s2.x, s1.y - s2.y)


class CityNetwork:
    """
    Manages the city's transit topology.
    
    A medium-sized city with 12 stops and 4 routes. Transfer stops allow
    passengers to switch between routes. Travel times between consecutive
    stops on a route are proportional to Euclidean distance.
    """

    def __init__(self) -> None:
        self.stops: Dict[str, Stop] = {}
        self.routes: Dict[str, Route] = {}
        self._travel_times: Dict[Tuple[str, str], int] = {}
        self._build_network()

    def _build_network(self) -> None:
        """Construct the default city network."""
        # Define 12 stops in a realistic grid-like layout
        stop_defs = [
            # id, name, x, y, base_demand, is_transfer
            ("S01", "Central Station", 5.0, 5.0, 8.0, True),
            ("S02", "Market Square", 3.0, 6.0, 6.0, False),
            ("S03", "University", 7.0, 7.0, 7.0, False),
            ("S04", "Hospital", 2.0, 4.0, 5.0, False),
            ("S05", "Tech Park", 8.0, 4.0, 9.0, True),
            ("S06", "Stadium", 4.0, 8.0, 4.0, False),
            ("S07", "Airport Road", 9.0, 6.0, 6.0, False),
            ("S08", "Old Town", 1.0, 5.0, 3.0, False),
            ("S09", "Business District", 6.0, 3.0, 8.0, True),
            ("S10", "Residential North", 5.0, 9.0, 5.0, False),
            ("S11", "Industrial Zone", 8.0, 1.0, 4.0, False),
            ("S12", "Suburb South", 3.0, 1.0, 3.0, False),
        ]

        for sid, name, x, y, demand, transfer in stop_defs:
            self.stops[sid] = Stop(
                stop_id=sid, name=name, x=x, y=y,
                base_demand=demand, is_transfer=transfer,
            )

        # Define 4 routes
        self.routes = {
            "R1": Route(
                route_id="R1",
                name="North-South Express",
                stop_ids=("S10", "S06", "S02", "S01", "S09", "S12"),
                base_frequency=3,
                color="#ef4444",
            ),
            "R2": Route(
                route_id="R2",
                name="East-West Connector",
                stop_ids=("S08", "S04", "S01", "S05", "S07"),
                base_frequency=3,
                color="#3b82f6",
            ),
            "R3": Route(
                route_id="R3",
                name="University Loop",
                stop_ids=("S01", "S03", "S05", "S09", "S01"),
                base_frequency=4,
                color="#10b981",
            ),
            "R4": Route(
                route_id="R4",
                name="Industrial Link",
                stop_ids=("S09", "S11", "S05", "S07"),
                base_frequency=5,
                color="#f59e0b",
            ),
        }

        # Precompute travel times (ticks) between consecutive stops on each route
        for route in self.routes.values():
            for i in range(len(route.stop_ids) - 1):
                s1 = self.stops[route.stop_ids[i]]
                s2 = self.stops[route.stop_ids[i + 1]]
                dist = _distance(s1, s2)
                # 1 tick per ~3 units of distance, minimum 1
                travel_time = max(1, round(dist / 3.0))
                key = (route.stop_ids[i], route.stop_ids[i + 1])
                self._travel_times[key] = travel_time

    def get_travel_time(self, from_stop: str, to_stop: str) -> int:
        """Get travel time in ticks between two consecutive stops."""
        key = (from_stop, to_stop)
        if key in self._travel_times:
            return self._travel_times[key]
        # Try reverse direction
        rev_key = (to_stop, from_stop)
        if rev_key in self._travel_times:
            return self._travel_times[rev_key]
        return 2  # default fallback

    def get_route_stops(self, route_id: str) -> List[Stop]:
        """Get ordered list of Stop objects for a route."""
        route = self.routes[route_id]
        return [self.stops[sid] for sid in route.stop_ids]

    def get_stop_routes(self, stop_id: str) -> List[str]:
        """Get all route IDs that serve a given stop."""
        result = []
        for route in self.routes.values():
            if stop_id in route.stop_ids:
                result.append(route.route_id)
        return result

    def get_transfer_stops(self) -> List[Stop]:
        """Get all transfer-capable stops."""
        return [s for s in self.stops.values() if s.is_transfer]

    def to_dict(self) -> dict:
        """Serialize the network for observation."""
        return {
            "stops": {
                sid: {
                    "name": s.name,
                    "x": s.x,
                    "y": s.y,
                    "base_demand": s.base_demand,
                    "is_transfer": s.is_transfer,
                }
                for sid, s in self.stops.items()
            },
            "routes": {
                rid: {
                    "name": r.name,
                    "stops": list(r.stop_ids),
                    "base_frequency": r.base_frequency,
                }
                for rid, r in self.routes.items()
            },
        }

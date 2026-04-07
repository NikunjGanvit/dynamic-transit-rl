"""
Simulation Engine — Core tick-based transit simulation.

Orchestrates the city network, buses, passengers, and events into a
coherent discrete-time simulation. Each tick represents ~5 minutes of
simulated time.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .bus import Bus, BusStatus, BusType, BUS_CAPACITY, create_default_fleet
from .city_network import CityNetwork
from .events import EventManager
from .passenger import PassengerGenerator


@dataclass
class SimulationMetrics:
    """Aggregated metrics for the current simulation state."""
    total_passengers_waiting: int = 0
    total_passengers_served: int = 0
    total_passengers_abandoned: int = 0
    avg_wait_time: float = 0.0
    avg_satisfaction: float = 1.0
    avg_occupancy: float = 0.0
    active_buses: int = 0
    idle_buses: int = 0
    max_queue_size: int = 0
    queue_variance: float = 0.0
    throughput: float = 0.0  # passengers served per tick

    def to_dict(self) -> dict:
        return {
            "total_waiting": self.total_passengers_waiting,
            "total_served": self.total_passengers_served,
            "total_abandoned": self.total_passengers_abandoned,
            "avg_wait_time": round(self.avg_wait_time, 2),
            "avg_satisfaction": round(self.avg_satisfaction, 3),
            "avg_occupancy_pct": round(self.avg_occupancy * 100, 1),
            "active_buses": self.active_buses,
            "idle_buses": self.idle_buses,
            "max_queue_size": self.max_queue_size,
            "queue_variance": round(self.queue_variance, 2),
            "throughput": round(self.throughput, 2),
        }


class SimulationEngine:
    """
    Core simulation engine for the transit environment.
    
    Manages the tick-based simulation loop:
    1. Generate new passengers
    2. Update passenger queues (patience/abandonment)
    3. Move buses along routes
    4. Board/alight passengers at stops
    5. Apply event effects
    6. Compute metrics
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.network = CityNetwork()
        self.event_manager = EventManager()
        self.passenger_gen = PassengerGenerator(self.network, self.rng)
        self.buses: Dict[str, Bus] = {}
        self.tick = 0
        self.max_ticks = 20
        self.done = False
        self._action_log: List[Dict[str, Any]] = []
        self._initial_metrics: Optional[SimulationMetrics] = None
        self._metrics_history: List[SimulationMetrics] = []
        self._last_action_error: Optional[str] = None

    def initialize(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the simulation with a task configuration.
        
        Args:
            task_config: Dict with keys:
                - seed: Random seed
                - max_ticks: Episode length
                - events: List of event configs
                - initial_queues: Optional dict of stop_id -> queue_size
                - fleet_overrides: Optional list of bus config overrides
                
        Returns:
            Initial observation dict
        """
        self.seed = task_config.get("seed", 42)
        self.rng = random.Random(self.seed)
        self.passenger_gen = PassengerGenerator(self.network, self.rng)
        self.max_ticks = task_config.get("max_ticks", 20)
        self.tick = 0
        self.done = False
        self._action_log = []
        self._metrics_history = []
        self._last_action_error = None

        # Set up fleet
        fleet = create_default_fleet()
        if "fleet_overrides" in task_config:
            for override in task_config["fleet_overrides"]:
                for bus in fleet:
                    if bus.bus_id == override["bus_id"]:
                        if "route_id" in override:
                            bus.route_id = override["route_id"]
                        if "position_idx" in override:
                            bus.position_idx = override["position_idx"]
                        if "status" in override:
                            bus.status = BusStatus(override["status"])
        self.buses = {b.bus_id: b for b in fleet}

        # Set up events
        self.event_manager.reset()
        if "events" in task_config:
            self.event_manager.schedule_events(task_config["events"])

        # Set up initial queues
        if "initial_queues" in task_config:
            for stop_id, count in task_config["initial_queues"].items():
                for i in range(count):
                    dest = self.passenger_gen._pick_destination(stop_id)
                    from .passenger import Passenger
                    p = Passenger(
                        passenger_id=self.passenger_gen._next_id,
                        origin_stop=stop_id,
                        destination_stop=dest,
                        arrival_tick=0,
                        patience=self.rng.randint(5, 15),
                    )
                    self.passenger_gen._next_id += 1
                    self.passenger_gen.queues[stop_id].append(p)
                    self.passenger_gen.total_generated += 1

        # Compute and store initial metrics
        self._initial_metrics = self._compute_metrics()
        self._metrics_history.append(self._initial_metrics)

        return self._build_observation()

    def step_simulation(self) -> None:
        """
        Advance the simulation by one tick.
        
        This is called automatically after each action.
        """
        if self.done:
            return

        self.tick += 1

        # 1. Get event modifiers for this tick
        demand_mults = self.event_manager.get_demand_multipliers(self.tick)
        time_mult = self.event_manager.get_time_multiplier(self.tick, self.max_ticks)
        broken_buses = self.event_manager.get_broken_buses(self.tick)

        # 2. Apply breakdown events
        for bus_id in broken_buses:
            if bus_id in self.buses:
                self.buses[bus_id].status = BusStatus.BREAKDOWN

        # 3. Generate new passengers
        self.passenger_gen.generate_passengers(
            self.tick, demand_mults, time_mult
        )

        # 4. Update queues (patience, abandonment)
        self.passenger_gen.update_queues(self.tick)

        # 5. Move buses and handle boarding/alighting
        for bus in self.buses.values():
            if bus.status == BusStatus.HELD:
                bus.hold_remaining -= 1
                if bus.hold_remaining <= 0:
                    bus.status = BusStatus.ACTIVE
                    bus.hold_remaining = 0
                continue

            if bus.status != BusStatus.ACTIVE or bus.route_id is None:
                continue

            route = self.network.routes.get(bus.route_id)
            if route is None:
                continue

            # Apply speed modifier from events
            speed_mult = self.event_manager.get_speed_multiplier(
                self.tick, bus.route_id
            )
            bus.speed_multiplier = speed_mult

            # Get current stop
            stop_ids = list(route.stop_ids)
            if bus.position_idx >= len(stop_ids):
                bus.position_idx = 0

            current_stop_id = stop_ids[bus.position_idx]

            # Alight passengers (random portion at each stop)
            if bus.passengers > 0:
                alight_count = max(1, self.rng.randint(
                    0, max(1, bus.passengers // 3)
                ))
                bus.alight_passengers(alight_count)

            # Board passengers from queue
            available = bus.available_seats
            if available > 0:
                queue_size = self.passenger_gen.get_queue_size(current_stop_id)
                board_count = min(available, queue_size)
                if board_count > 0:
                    self.passenger_gen.board_from_stop(
                        current_stop_id, board_count
                    )
                    bus.board_passengers(board_count)

            # Move to next stop
            bus.travel_progress += 1
            next_idx = bus.position_idx + bus.direction
            if next_idx >= len(stop_ids):
                bus.direction = -1
                next_idx = bus.position_idx - 1
            elif next_idx < 0:
                bus.direction = 1
                next_idx = 1

            # Check if travel time elapsed
            next_stop_id = stop_ids[max(0, min(next_idx, len(stop_ids) - 1))]
            travel_time = self.network.get_travel_time(
                current_stop_id, next_stop_id
            )
            adjusted_time = max(1, round(travel_time / max(0.1, speed_mult)))

            if bus.travel_progress >= adjusted_time:
                bus.position_idx = max(0, min(next_idx, len(stop_ids) - 1))
                bus.travel_progress = 0

        # 6. Check if episode is done
        if self.tick >= self.max_ticks:
            self.done = True

        # 7. Record metrics
        metrics = self._compute_metrics()
        self._metrics_history.append(metrics)

    # ─── Action Methods ─────────────────────────────────────────

    def action_reassign_bus(self, bus_id: str, target_route_id: str) -> str:
        """Reassign a bus to a different route."""
        self._last_action_error = None

        if bus_id not in self.buses:
            self._last_action_error = f"Bus {bus_id} not found"
            return self._last_action_error

        bus = self.buses[bus_id]
        if bus.status == BusStatus.BREAKDOWN:
            self._last_action_error = f"Bus {bus_id} is broken down"
            return self._last_action_error

        if target_route_id not in self.network.routes:
            self._last_action_error = f"Route {target_route_id} not found"
            return self._last_action_error

        old_route = bus.route_id
        bus.route_id = target_route_id
        bus.position_idx = 0
        bus.travel_progress = 0
        bus.direction = 1
        bus.status = BusStatus.ACTIVE
        # Passengers are "transferred" - they stay on the bus
        self._action_log.append({
            "tick": self.tick,
            "action": "reassign_bus",
            "bus_id": bus_id,
            "from_route": old_route,
            "to_route": target_route_id,
        })
        return f"Bus {bus_id} reassigned from {old_route} to {target_route_id}"

    def action_dispatch_bus(self, route_id: str, bus_type: str = "standard") -> str:
        """Dispatch a bus from depot to a route."""
        self._last_action_error = None

        if route_id not in self.network.routes:
            self._last_action_error = f"Route {route_id} not found"
            return self._last_action_error

        try:
            bt = BusType(bus_type.lower())
        except ValueError:
            self._last_action_error = f"Invalid bus type: {bus_type}. Use: standard, articulated, mini"
            return self._last_action_error

        # Find a depot bus or create one
        depot_bus = None
        for bus in self.buses.values():
            if bus.status == BusStatus.DEPOT:
                depot_bus = bus
                break

        if depot_bus is None:
            # Create a new bus if fleet allows (max 12)
            if len(self.buses) >= 12:
                self._last_action_error = "Fleet limit reached (12 buses max)"
                return self._last_action_error
            new_id = f"B{len(self.buses) + 1:02d}"
            depot_bus = Bus(
                bus_id=new_id,
                bus_type=bt,
                route_id=route_id,
                position_idx=0,
                status=BusStatus.ACTIVE,
            )
            self.buses[new_id] = depot_bus
        else:
            depot_bus.route_id = route_id
            depot_bus.position_idx = 0
            depot_bus.status = BusStatus.ACTIVE
            depot_bus.bus_type = bt
            depot_bus.capacity = BUS_CAPACITY[bt]
            depot_bus.passengers = 0
            depot_bus.direction = 1
            depot_bus.travel_progress = 0

        self._action_log.append({
            "tick": self.tick,
            "action": "dispatch_bus",
            "bus_id": depot_bus.bus_id,
            "route_id": route_id,
            "bus_type": bus_type,
        })
        return f"Bus {depot_bus.bus_id} ({bt.value}) dispatched to route {route_id}"

    def action_increase_frequency(self, route_id: str) -> str:
        """Increase frequency by speeding up all buses on a route."""
        self._last_action_error = None

        if route_id not in self.network.routes:
            self._last_action_error = f"Route {route_id} not found"
            return self._last_action_error

        count = 0
        for bus in self.buses.values():
            if bus.route_id == route_id and bus.status == BusStatus.ACTIVE:
                bus.speed_multiplier = min(2.0, bus.speed_multiplier * 1.3)
                count += 1

        if count == 0:
            self._last_action_error = f"No active buses on route {route_id}"
            return self._last_action_error

        self._action_log.append({
            "tick": self.tick,
            "action": "increase_frequency",
            "route_id": route_id,
            "buses_affected": count,
        })
        return f"Increased frequency on route {route_id} ({count} buses sped up by 30%)"

    def action_hold_bus(self, bus_id: str, duration: int = 2) -> str:
        """Hold a bus at current stop for N ticks."""
        self._last_action_error = None

        if bus_id not in self.buses:
            self._last_action_error = f"Bus {bus_id} not found"
            return self._last_action_error

        bus = self.buses[bus_id]
        if bus.status != BusStatus.ACTIVE:
            self._last_action_error = f"Bus {bus_id} is not active (status: {bus.status.value})"
            return self._last_action_error

        duration = max(1, min(duration, 5))  # clamp to 1-5
        bus.status = BusStatus.HELD
        bus.hold_remaining = duration

        self._action_log.append({
            "tick": self.tick,
            "action": "hold_bus",
            "bus_id": bus_id,
            "duration": duration,
        })
        return f"Bus {bus_id} held at current stop for {duration} ticks"

    def action_skip(self) -> str:
        """No-op action."""
        self._action_log.append({
            "tick": self.tick,
            "action": "skip",
        })
        return "No action taken. Simulation advanced by one tick."

    # ─── Metrics & Observation ────────────────────────────────

    def _compute_metrics(self) -> SimulationMetrics:
        """Compute current simulation metrics."""
        queue_sizes = []
        for stop_id in self.network.stops:
            queue_sizes.append(self.passenger_gen.get_queue_size(stop_id))

        total_waiting = sum(queue_sizes)
        max_queue = max(queue_sizes) if queue_sizes else 0
        mean_queue = total_waiting / len(queue_sizes) if queue_sizes else 0
        variance = (
            sum((q - mean_queue) ** 2 for q in queue_sizes) / len(queue_sizes)
            if queue_sizes else 0.0
        )

        active_buses = sum(
            1 for b in self.buses.values()
            if b.status == BusStatus.ACTIVE
        )
        idle_buses = sum(
            1 for b in self.buses.values()
            if b.status in (BusStatus.DEPOT, BusStatus.HELD)
        )

        occupancies = [
            b.occupancy_ratio for b in self.buses.values()
            if b.status == BusStatus.ACTIVE and b.route_id is not None
        ]
        avg_occ = sum(occupancies) / len(occupancies) if occupancies else 0.0

        total_served = self.passenger_gen.total_boarded
        throughput = total_served / max(1, self.tick) if self.tick > 0 else 0.0

        return SimulationMetrics(
            total_passengers_waiting=total_waiting,
            total_passengers_served=total_served,
            total_passengers_abandoned=self.passenger_gen.total_abandoned,
            avg_wait_time=self.passenger_gen.get_avg_wait_time(),
            avg_satisfaction=self.passenger_gen.get_avg_satisfaction(),
            avg_occupancy=avg_occ,
            active_buses=active_buses,
            idle_buses=idle_buses,
            max_queue_size=max_queue,
            queue_variance=variance,
            throughput=throughput,
        )

    def get_current_metrics(self) -> SimulationMetrics:
        """Get current simulation metrics."""
        return self._compute_metrics()

    def get_initial_metrics(self) -> Optional[SimulationMetrics]:
        """Get initial metrics at tick 0."""
        return self._initial_metrics

    def _build_observation(self) -> Dict[str, Any]:
        """Build the full observation dict for the agent."""
        metrics = self._compute_metrics()
        queue_stats = self.passenger_gen.get_queue_stats()
        active_events = self.event_manager.get_active_event_descriptions(self.tick)

        return {
            "tick": self.tick,
            "max_ticks": self.max_ticks,
            "done": self.done,
            "metrics": metrics.to_dict(),
            "stops": {
                sid: {
                    "name": self.network.stops[sid].name,
                    "queue_size": queue_stats[sid]["queue_size"],
                    "avg_wait": round(queue_stats[sid]["avg_wait"], 1),
                    "max_wait": queue_stats[sid]["max_wait"],
                    "base_demand": self.network.stops[sid].base_demand,
                }
                for sid in self.network.stops
            },
            "buses": {
                bid: bus.to_dict()
                for bid, bus in self.buses.items()
            },
            "routes": {
                rid: {
                    "name": r.name,
                    "stops": list(r.stop_ids),
                    "buses_on_route": [
                        b.bus_id for b in self.buses.values()
                        if b.route_id == rid and b.status == BusStatus.ACTIVE
                    ],
                    "total_queue_at_stops": sum(
                        self.passenger_gen.get_queue_size(sid)
                        for sid in r.stop_ids
                    ),
                }
                for rid, r in self.network.routes.items()
            },
            "active_events": active_events,
            "last_action_error": self._last_action_error,
        }

    def get_observation(self) -> Dict[str, Any]:
        """Public accessor for current observation."""
        return self._build_observation()

    def get_state_dict(self) -> Dict[str, Any]:
        """Get full internal state for state() endpoint."""
        return {
            "tick": self.tick,
            "max_ticks": self.max_ticks,
            "done": self.done,
            "seed": self.seed,
            "total_generated": self.passenger_gen.total_generated,
            "total_boarded": self.passenger_gen.total_boarded,
            "total_abandoned": self.passenger_gen.total_abandoned,
            "action_count": len(self._action_log),
            "action_log": self._action_log[-5:],  # last 5 actions
        }

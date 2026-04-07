"""
Passenger — Passenger generation and queue management.

Generates passengers at stops using Poisson-distributed arrivals,
with demand modulated by time-of-day and events.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .city_network import CityNetwork, Stop


@dataclass
class Passenger:
    """A passenger waiting at a stop or riding a bus."""
    passenger_id: int
    origin_stop: str
    destination_stop: str
    arrival_tick: int
    patience: int  # max ticks willing to wait
    wait_time: int = 0
    boarded: bool = False
    abandoned: bool = False

    @property
    def satisfaction(self) -> float:
        """
        Satisfaction score based on wait time relative to patience.
        1.0 = immediate boarding, 0.0 = waited full patience.
        """
        if self.patience == 0:
            return 1.0
        ratio = min(self.wait_time / self.patience, 1.0)
        return max(0.0, 1.0 - ratio)


class PassengerGenerator:
    """
    Generates passengers at stops with demand modulated by events.
    
    Uses Poisson process with rate = base_demand * time_multiplier * event_multiplier.
    Passenger destinations are chosen from stops on the same route or transfer stops.
    """

    def __init__(self, network: CityNetwork, rng: random.Random) -> None:
        self.network = network
        self.rng = rng
        self._next_id = 1
        # Queues: stop_id -> list of passengers waiting
        self.queues: Dict[str, List[Passenger]] = {
            sid: [] for sid in network.stops
        }
        # Tracking metrics
        self.total_generated = 0
        self.total_abandoned = 0
        self.total_boarded = 0
        self.satisfaction_scores: List[float] = []

    def generate_passengers(
        self,
        current_tick: int,
        event_multipliers: Dict[str, float],
        time_multiplier: float = 1.0,
    ) -> Dict[str, int]:
        """
        Generate new passengers at each stop for this tick.
        
        Args:
            current_tick: Current simulation tick
            event_multipliers: Per-stop demand multipliers from events
            time_multiplier: Global time-of-day multiplier
            
        Returns:
            Dict mapping stop_id to number of new passengers generated
        """
        generated = {}
        for stop_id, stop in self.network.stops.items():
            rate = stop.base_demand * time_multiplier
            rate *= event_multipliers.get(stop_id, 1.0)
            # Poisson-distributed arrivals
            count = self._poisson(rate)
            generated[stop_id] = count

            for _ in range(count):
                dest = self._pick_destination(stop_id)
                patience = self.rng.randint(3, 12)  # 15-60 min patience
                passenger = Passenger(
                    passenger_id=self._next_id,
                    origin_stop=stop_id,
                    destination_stop=dest,
                    arrival_tick=current_tick,
                    patience=patience,
                )
                self._next_id += 1
                self.queues[stop_id].append(passenger)
                self.total_generated += 1

        return generated

    def update_queues(self, current_tick: int) -> Dict[str, int]:
        """
        Update wait times and remove abandoned passengers.
        
        Returns:
            Dict mapping stop_id to number of passengers who abandoned
        """
        abandonments = {}
        for stop_id, queue in self.queues.items():
            abandoned_count = 0
            remaining = []
            for p in queue:
                p.wait_time += 1
                if p.wait_time >= p.patience:
                    p.abandoned = True
                    abandoned_count += 1
                    self.total_abandoned += 1
                    self.satisfaction_scores.append(0.0)  # worst satisfaction
                else:
                    remaining.append(p)
            self.queues[stop_id] = remaining
            abandonments[stop_id] = abandoned_count
        return abandonments

    def board_from_stop(self, stop_id: str, count: int) -> List[Passenger]:
        """
        Board up to `count` passengers from a stop's queue.
        Returns list of boarded passengers (FIFO order).
        """
        queue = self.queues[stop_id]
        boarded = queue[:count]
        self.queues[stop_id] = queue[count:]
        for p in boarded:
            p.boarded = True
            self.total_boarded += 1
            self.satisfaction_scores.append(p.satisfaction)
        return boarded

    def get_queue_size(self, stop_id: str) -> int:
        """Get current queue size at a stop."""
        return len(self.queues[stop_id])

    def get_total_waiting(self) -> int:
        """Total passengers waiting across all stops."""
        return sum(len(q) for q in self.queues.values())

    def get_avg_wait_time(self) -> float:
        """Average wait time across all currently waiting passengers."""
        all_waits = []
        for queue in self.queues.values():
            for p in queue:
                all_waits.append(p.wait_time)
        return sum(all_waits) / len(all_waits) if all_waits else 0.0

    def get_avg_satisfaction(self) -> float:
        """Average satisfaction across all boarded/abandoned passengers."""
        if not self.satisfaction_scores:
            return 1.0
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores)

    def _poisson(self, lam: float) -> int:
        """Generate a Poisson-distributed random number."""
        # Use inverse transform method for small lambda
        if lam <= 0:
            return 0
        L = 2.71828 ** (-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rng.random()
            if p <= L:
                return k - 1

    def _pick_destination(self, origin_stop: str) -> str:
        """Pick a random destination stop (different from origin)."""
        candidates = [
            sid for sid in self.network.stops
            if sid != origin_stop
        ]
        if not candidates:
            return origin_stop
        return self.rng.choice(candidates)

    def get_queue_stats(self) -> dict:
        """Get queue statistics for all stops."""
        stats = {}
        for stop_id in self.network.stops:
            queue = self.queues[stop_id]
            stats[stop_id] = {
                "queue_size": len(queue),
                "avg_wait": (
                    sum(p.wait_time for p in queue) / len(queue)
                    if queue else 0.0
                ),
                "max_wait": max((p.wait_time for p in queue), default=0),
            }
        return stats

    def reset(self) -> None:
        """Reset all queues and counters."""
        self.queues = {sid: [] for sid in self.network.stops}
        self._next_id = 1
        self.total_generated = 0
        self.total_abandoned = 0
        self.total_boarded = 0
        self.satisfaction_scores = []

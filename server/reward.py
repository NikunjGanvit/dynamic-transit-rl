"""
Reward Function — Composite reward with partial progress signals.

Provides continuous reward signal based on multiple transit efficiency
metrics. Rewards good decisions and penalizes harmful ones.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .simulation.engine import SimulationEngine, SimulationMetrics


class RewardCalculator:
    """
    Computes a composite reward in range ~[-0.10, 1.0].
    
    Components:
    - Wait time score (30%): Lower avg wait → higher reward
    - Occupancy score (25%): Optimal occupancy ~60-80% → higher reward
    - Throughput score (20%): More passengers served → higher reward
    - Satisfaction score (15%): Higher passenger satisfaction → higher reward
    - Idle penalty (5%): Fewer idle buses → less penalty
    - Overcrowding penalty (5%): Fewer overcrowded stops → less penalty
    """

    WAIT_WEIGHT = 0.30
    OCCUPANCY_WEIGHT = 0.25
    THROUGHPUT_WEIGHT = 0.20
    SATISFACTION_WEIGHT = 0.15
    IDLE_PENALTY_WEIGHT = 0.05
    OVERCROWDING_PENALTY_WEIGHT = 0.05

    def __init__(self) -> None:
        self._prev_metrics: Optional[SimulationMetrics] = None
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0

    def compute_reward(
        self,
        engine: SimulationEngine,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute reward for the current simulation state.
        
        Args:
            engine: The simulation engine
            task_weights: Optional per-task weight overrides
            
        Returns:
            Reward value in ~[-0.10, 1.0]
        """
        metrics = engine.get_current_metrics()
        self._step_count += 1

        # --- Wait Time Score ---
        # Normalize: 0 wait = 1.0, 10+ ticks avg wait = 0.0
        max_acceptable_wait = 10.0
        wait_score = max(0.0, 1.0 - (metrics.avg_wait_time / max_acceptable_wait))

        # --- Occupancy Score ---
        # Optimal occupancy is 60-80%. Too low = waste, too high = overcrowded
        occ = metrics.avg_occupancy
        if 0.5 <= occ <= 0.85:
            occupancy_score = 1.0  # sweet spot
        elif occ < 0.5:
            occupancy_score = occ / 0.5  # linearly scale up
        else:
            occupancy_score = max(0.0, 1.0 - (occ - 0.85) / 0.15)  # penalize overcrowding

        # --- Throughput Score ---
        # Based on ratio of passengers served to total generated
        total_gen = engine.passenger_gen.total_generated
        if total_gen > 0:
            throughput_score = min(1.0, metrics.total_passengers_served / total_gen)
        else:
            throughput_score = 1.0  # no passengers = perfect

        # --- Satisfaction Score ---
        satisfaction_score = metrics.avg_satisfaction

        # --- Idle Penalty ---
        total_buses = len(engine.buses)
        if total_buses > 0:
            idle_ratio = metrics.idle_buses / total_buses
        else:
            idle_ratio = 0.0
        idle_penalty = idle_ratio

        # --- Overcrowding Penalty ---
        # Count stops with queue > 15
        overcrowded_count = 0
        for stop_id in engine.network.stops:
            if engine.passenger_gen.get_queue_size(stop_id) > 15:
                overcrowded_count += 1
        overcrowding_penalty = overcrowded_count / max(1, len(engine.network.stops))

        # --- Composite Reward ---
        weights = task_weights or {}
        w_wait = weights.get("wait", self.WAIT_WEIGHT)
        w_occ = weights.get("occupancy", self.OCCUPANCY_WEIGHT)
        w_thru = weights.get("throughput", self.THROUGHPUT_WEIGHT)
        w_sat = weights.get("satisfaction", self.SATISFACTION_WEIGHT)
        w_idle = weights.get("idle_penalty", self.IDLE_PENALTY_WEIGHT)
        w_crowd = weights.get("overcrowding_penalty", self.OVERCROWDING_PENALTY_WEIGHT)

        reward = (
            w_wait * wait_score
            + w_occ * occupancy_score
            + w_thru * throughput_score
            + w_sat * satisfaction_score
            - w_idle * idle_penalty
            - w_crowd * overcrowding_penalty
        )

        # Clamp to [-0.10, 1.0]
        reward = max(-0.10, min(1.0, reward))

        self._prev_metrics = metrics
        self._cumulative_reward += reward

        return round(reward, 4)

    def get_cumulative_reward(self) -> float:
        """Get total accumulated reward."""
        return round(self._cumulative_reward, 4)

    def get_average_reward(self) -> float:
        """Get average reward per step."""
        if self._step_count == 0:
            return 0.0
        return round(self._cumulative_reward / self._step_count, 4)

    def reset(self) -> None:
        """Reset reward calculator."""
        self._prev_metrics = None
        self._cumulative_reward = 0.0
        self._step_count = 0

"""
Transit Environment — MCPEnvironment implementation for OpenEnv.

This is the main environment class that exposes the transit simulation
as MCP tools, following the OpenEnv specification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .simulation.engine import SimulationEngine
from .tasks import TASK_REGISTRY
from .graders import GRADER_REGISTRY
from .reward import RewardCalculator


class TransitEnvironment(MCPEnvironment):
    """
    Urban Transit Operations Environment.
    
    An OpenEnv-compliant environment that simulates urban public transport
    management. The agent acts as a transit operations manager, making
    decisions to optimize system performance under dynamic conditions.
    
    MCP Tools:
    - get_system_status: View full system state
    - get_route_analytics: Detailed analytics for a specific route
    - get_task_info: Current task objective and progress
    - reassign_bus: Move a bus to a different route
    - dispatch_bus: Deploy a bus from depot to a route
    - increase_frequency: Speed up buses on a route
    - hold_bus: Hold a bus at current stop
    - skip_action: Do nothing (advance simulation)
    """

    def __init__(self) -> None:
        """Initialize the transit environment with MCP server and tools."""
        mcp = FastMCP("transit_env")

        # Store reference for tool closures
        env_ref = self

        # ─── Observation Tools ────────────────────────────────

        @mcp.tool
        def get_system_status() -> str:
            """
            Get the complete system status including all stops, buses,
            routes, queue sizes, active events, and performance metrics.
            Call this at the start of each step to understand the current state.
            
            Returns:
                JSON string with full system observation
            """
            obs = env_ref._engine.get_observation()
            return json.dumps(obs, indent=2)

        @mcp.tool
        def get_route_analytics(route_id: str) -> str:
            """
            Get detailed analytics for a specific route.
            
            Args:
                route_id: Route identifier (R1, R2, R3, or R4)
                
            Returns:
                JSON string with route-specific analytics
            """
            if route_id not in env_ref._engine.network.routes:
                return json.dumps({"error": f"Route {route_id} not found. Valid: R1, R2, R3, R4"})
            
            route = env_ref._engine.network.routes[route_id]
            buses_on_route = [
                b.to_dict() for b in env_ref._engine.buses.values()
                if b.route_id == route_id
            ]
            stop_queues = {}
            for sid in route.stop_ids:
                stop_queues[sid] = {
                    "name": env_ref._engine.network.stops[sid].name,
                    "queue_size": env_ref._engine.passenger_gen.get_queue_size(sid),
                    "base_demand": env_ref._engine.network.stops[sid].base_demand,
                }
            
            analytics = {
                "route_id": route_id,
                "route_name": route.name,
                "stops": list(route.stop_ids),
                "stop_details": stop_queues,
                "buses": buses_on_route,
                "total_buses": len(buses_on_route),
                "total_queue": sum(q["queue_size"] for q in stop_queues.values()),
            }
            return json.dumps(analytics, indent=2)

        @mcp.tool
        def get_task_info() -> str:
            """
            Get the current task objective, difficulty, and progress.
            Call this at the beginning to understand what you need to accomplish.
            
            Returns:
                JSON string with task information and current score
            """
            info = {
                "task": env_ref._current_task.to_dict() if env_ref._current_task else None,
                "current_tick": env_ref._engine.tick,
                "max_ticks": env_ref._engine.max_ticks,
                "current_reward": env_ref._reward_calc.get_cumulative_reward(),
                "avg_reward_per_step": env_ref._reward_calc.get_average_reward(),
            }
            return json.dumps(info, indent=2)

        # ─── Action Tools ────────────────────────────────────

        @mcp.tool
        def reassign_bus(bus_id: str, target_route_id: str) -> str:
            """
            Reassign a bus to a different route. The bus will start at the
            first stop of the new route. Existing passengers stay on board.
            
            Args:
                bus_id: Bus identifier (e.g., B01, B02, ..., B08)
                target_route_id: Target route (R1, R2, R3, or R4)
                
            Returns:
                Result message describing what happened
            """
            result = env_ref._engine.action_reassign_bus(bus_id, target_route_id)
            env_ref._advance_and_compute()
            obs = env_ref._engine.get_observation()
            return json.dumps({
                "action_result": result,
                "observation": obs,
                "reward": env_ref._last_reward,
                "done": env_ref._engine.done,
            }, indent=2)

        @mcp.tool
        def dispatch_bus(route_id: str, bus_type: str = "standard") -> str:
            """
            Dispatch a bus from depot to a route, or create a new bus if
            no depot buses are available (max fleet: 12 buses).
            
            Args:
                route_id: Target route (R1, R2, R3, or R4)
                bus_type: Type of bus: 'standard' (cap=40), 'articulated' (cap=80), 'mini' (cap=20)
                
            Returns:
                Result message describing what happened
            """
            result = env_ref._engine.action_dispatch_bus(route_id, bus_type)
            env_ref._advance_and_compute()
            obs = env_ref._engine.get_observation()
            return json.dumps({
                "action_result": result,
                "observation": obs,
                "reward": env_ref._last_reward,
                "done": env_ref._engine.done,
            }, indent=2)

        @mcp.tool
        def increase_frequency(route_id: str) -> str:
            """
            Increase service frequency on a route by speeding up all active
            buses on that route by 30%. Can be called multiple times (caps at 2x).
            
            Args:
                route_id: Target route (R1, R2, R3, or R4)
                
            Returns:
                Result message describing what happened
            """
            result = env_ref._engine.action_increase_frequency(route_id)
            env_ref._advance_and_compute()
            obs = env_ref._engine.get_observation()
            return json.dumps({
                "action_result": result,
                "observation": obs,
                "reward": env_ref._last_reward,
                "done": env_ref._engine.done,
            }, indent=2)

        @mcp.tool
        def hold_bus(bus_id: str, duration: int = 2) -> str:
            """
            Hold a bus at its current stop for extra boarding time.
            Useful when a stop has a large queue and you want to board more passengers.
            
            Args:
                bus_id: Bus identifier (e.g., B01, B02)
                duration: Number of ticks to hold (1-5)
                
            Returns:
                Result message describing what happened
            """
            result = env_ref._engine.action_hold_bus(bus_id, duration)
            env_ref._advance_and_compute()
            obs = env_ref._engine.get_observation()
            return json.dumps({
                "action_result": result,
                "observation": obs,
                "reward": env_ref._last_reward,
                "done": env_ref._engine.done,
            }, indent=2)

        @mcp.tool
        def skip_action() -> str:
            """
            Skip this step without taking any action. The simulation 
            still advances by one tick.
            
            Returns:
                Result message with updated observation
            """
            result = env_ref._engine.action_skip()
            env_ref._advance_and_compute()
            obs = env_ref._engine.get_observation()
            return json.dumps({
                "action_result": result,
                "observation": obs,
                "reward": env_ref._last_reward,
                "done": env_ref._engine.done,
            }, indent=2)

        # Pass the MCP server to the base class
        super().__init__(mcp)

        # Internal state
        self._engine = SimulationEngine()
        self._reward_calc = RewardCalculator()
        self._current_task = None
        self._current_task_name = "reduce_overcrowding"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_reward = 0.0
        self._rewards: List[float] = []

    def _advance_and_compute(self) -> None:
        """Advance simulation by one tick and compute reward."""
        self._engine.step_simulation()
        task_config = self._current_task.get_config() if self._current_task else {}
        reward_weights = task_config.get("reward_weights")
        self._last_reward = self._reward_calc.compute_reward(
            self._engine, reward_weights
        )
        self._rewards.append(self._last_reward)
        # self._state.step_count += 1  # REMOVED: Managed by step() / step_async()


    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment with a task configuration.
        
        Accepts optional task_name in kwargs to select which task to run.
        Default: reduce_overcrowding (easy).
        
        Args:
            seed: Optional random seed override
            episode_id: Optional episode ID
            **kwargs: May include 'task_name' to select task
            
        Returns:
            Initial observation
        """
        # Select task
        task_name = kwargs.get("task_name", self._current_task_name)
        if task_name in TASK_REGISTRY:
            self._current_task_name = task_name
            self._current_task = TASK_REGISTRY[task_name]()
        else:
            self._current_task_name = "reduce_overcrowding"
            self._current_task = TASK_REGISTRY["reduce_overcrowding"]()

        # Get task config and apply seed override
        config = self._current_task.get_config()
        if seed is not None:
            config["seed"] = seed

        # Reset engine and reward
        self._engine = SimulationEngine(seed=config.get("seed", 42))
        self._reward_calc = RewardCalculator()
        self._rewards = []
        self._last_reward = 0.0

        # Initialize simulation
        obs_dict = self._engine.initialize(config)

        # Reset state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Build initial observation message
        task_info = self._current_task.to_dict()
        objective = self._current_task.get_objective_prompt()

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": task_info,
                "objective": objective,
                "observation": obs_dict,
                "message": (
                    f"Transit environment reset for task: {task_info['task_name']} "
                    f"({task_info['difficulty']}). Use get_task_info() to see your objective, "
                    f"then get_system_status() to see the current state."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (returns error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use MCP tools: get_system_status, reassign_bus, dispatch_bus, "
                    "increase_frequency, hold_bus, skip_action."
                ),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.
        
        MCP actions are handled by the base class (tool calls).
        The simulation advances when action tools are called.
        """
        # Only increment step_count for actions that advance time
        is_time_advancing = False
        if hasattr(action, "tool_name") and action.tool_name in ["reassign_bus", "dispatch_bus", "increase_frequency", "hold_bus", "skip_action"]:
            is_time_advancing = True
        
        if is_time_advancing:
            self._state.step_count += 1


        # Delegate to base class for MCP routing
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Add reward and done info to observation
        if obs.metadata is None:
            obs.metadata = {}
        obs.reward = self._last_reward
        obs.done = self._engine.done

        # If episode is done, compute final grade
        if self._engine.done and self._current_task_name in GRADER_REGISTRY:
            grader = GRADER_REGISTRY[self._current_task_name]()
            final_score = grader.grade(self._engine)
            obs.metadata["final_score"] = final_score
            obs.metadata["final_metrics"] = self._engine.get_current_metrics().to_dict()
            obs.metadata["cumulative_reward"] = self._reward_calc.get_cumulative_reward()

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by WebSocket handler."""
        is_time_advancing = False
        if hasattr(action, "tool_name") and action.tool_name in ["reassign_bus", "dispatch_bus", "increase_frequency", "hold_bus", "skip_action"]:
            is_time_advancing = True

        if is_time_advancing:
            self._state.step_count += 1

        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        
        if obs.metadata is None:
            obs.metadata = {}
        obs.reward = self._last_reward
        obs.done = self._engine.done

        if self._engine.done and self._current_task_name in GRADER_REGISTRY:
            grader = GRADER_REGISTRY[self._current_task_name]()
            final_score = grader.grade(self._engine)
            obs.metadata["final_score"] = final_score
            obs.metadata["final_metrics"] = self._engine.get_current_metrics().to_dict()
            obs.metadata["cumulative_reward"] = self._reward_calc.get_cumulative_reward()

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_graded_score(self) -> float:
        """Get the final graded score for the current task."""
        if self._current_task_name in GRADER_REGISTRY:
            grader = GRADER_REGISTRY[self._current_task_name]()
            return grader.grade(self._engine)
        return 0.0

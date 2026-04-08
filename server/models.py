"""
Typed Pydantic models for the Transit Environment.
These satisfy the OpenEnv requirement for typed Action, Observation, and Reward models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class BusState(BaseModel):
    """Current state of a single bus."""
    bus_id: str
    type: str
    route_id: Optional[str]
    stop_idx: int
    passengers: int
    capacity: int
    status: str


class StopState(BaseModel):
    """Current state of a bus stop."""
    stop_id: str
    name: str
    queue_size: int
    base_demand: float


class TransitObservation(BaseModel):
    """Full system observation."""
    tick: int
    max_ticks: int
    done: bool
    buses: List[BusState]
    stops: Dict[str, StopState]
    active_events: List[str]
    metrics: Dict[str, Any]


class ActionCall(BaseModel):
    """A generic tool call action following the MCP pattern."""
    tool: str
    args: Dict[str, Any]


class RewardModel(BaseModel):
    """Reward breakdown for a single step."""
    total: float
    components: Dict[str, float]

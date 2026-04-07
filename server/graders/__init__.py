from .base import BaseGrader
from .overcrowding import OvercrowdingGrader
from .load_balance import LoadBalanceGrader
from .demand_spike import DemandSpikeGrader
from .multi_objective import MultiObjectiveGrader

GRADER_REGISTRY = {
    "reduce_overcrowding": OvercrowdingGrader,
    "balance_load": LoadBalanceGrader,
    "demand_spike": DemandSpikeGrader,
    "multi_objective": MultiObjectiveGrader,
}

__all__ = [
    "BaseGrader",
    "OvercrowdingGrader",
    "LoadBalanceGrader",
    "DemandSpikeGrader",
    "MultiObjectiveGrader",
    "GRADER_REGISTRY",
]

from .base import BaseTask
from .task_overcrowding import OvercrowdingTask
from .task_load_balance import LoadBalanceTask
from .task_demand_spike import DemandSpikeTask
from .task_multi_obj import MultiObjectiveTask

TASK_REGISTRY = {
    "reduce_overcrowding": OvercrowdingTask,
    "balance_load": LoadBalanceTask,
    "demand_spike": DemandSpikeTask,
    "multi_objective": MultiObjectiveTask,
}

__all__ = [
    "BaseTask",
    "OvercrowdingTask",
    "LoadBalanceTask",
    "DemandSpikeTask",
    "MultiObjectiveTask",
    "TASK_REGISTRY",
]

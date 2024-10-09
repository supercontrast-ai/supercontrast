from typing import List, Optional

from supercontrast.optimizer.handlers.mock import OptimizerMockHandler
from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.optimizer.optimizer_handler import OptimizerHandler
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task


def optimizer_factory(
    task: Task,
    providers: List[Provider],
    optimizer: Optional[Optimizer] = None,
) -> OptimizerHandler:
    if optimizer is None:
        return OptimizerMockHandler(task, providers)
    elif optimizer == Optimizer.LATENCY:
        return OptimizerMockHandler(task, providers)
    elif optimizer == Optimizer.COST:
        return OptimizerMockHandler(task, providers)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

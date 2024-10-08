from typing import List, Optional

from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.task_handler import TaskHandler


def task_factory(
    task: Task,
    providers: List[Provider],
    optimizer: Optional[Optimizer] = None,
    **config,
):
    return TaskHandler(task, providers, optimizer, **config)

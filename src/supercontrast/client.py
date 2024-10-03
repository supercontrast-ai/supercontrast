from typing import List, Optional

from supercontrast.optimizer import Optimizer
from supercontrast.provider import Provider
from supercontrast.task import Task, TaskHandler
from supercontrast.task.task_factory import task_factory


def supercontrast_client(
    task: Task,
    providers: List[Provider],
    optimizer: Optional[Optimizer] = None,
    **config
) -> TaskHandler:
    task_handler = task_factory(task, providers, optimizer, **config)
    return task_handler
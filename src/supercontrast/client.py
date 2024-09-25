from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.provider import Provider
from supercontrast.task import Task
from supercontrast.task.task_factory import task_factory


def supercontrast_client(
    task: Task,
    providers: List[Provider],
    optimize_by: Optional[OptimizerFunction] = None,
    **config
):
    task_handler = task_factory(task, providers, optimize_by, **config)
    return task_handler

from typing import List, Optional

from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.task_factory import task_factory
from supercontrast.task.task_handler import TaskHandler


class SuperContrastClient:
    def __init__(
        self,
        task: Task,
        providers: List[Provider],
        optimizer: Optional[Optimizer] = None,
        **config
    ):
        self.task_handler: TaskHandler = task_factory(
            task, providers, optimizer, **config
        )

    def request(self, *args, **kwargs):
        return self.task_handler.request(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.task_handler.evaluate(*args, **kwargs)

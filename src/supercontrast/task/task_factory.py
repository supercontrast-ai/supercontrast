from typing import List, Optional

from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_factory import get_supported_tasks_for_provider
from supercontrast.task.task_enum import Task
from supercontrast.task.task_handler import TaskHandler


def get_supported_providers_for_task(task: Task) -> List[Provider]:
    return [
        provider
        for provider in Provider
        if task in get_supported_tasks_for_provider(provider)
    ]


def task_factory(
    task: Task,
    providers: List[Provider],
    optimizer: Optional[Optimizer] = None,
    **config,
):
    return TaskHandler(task, providers, optimizer, **config)

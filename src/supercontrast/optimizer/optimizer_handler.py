from abc import ABC, abstractmethod
from typing import List

from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.provider import Provider
from supercontrast.task import Task


class OptimizerHandler(ABC):
    def __init__(self, task: Task, providers: List[Provider]):
        self.task = task
        self.providers = providers

    @abstractmethod
    def get_provider(self) -> Provider:
        pass

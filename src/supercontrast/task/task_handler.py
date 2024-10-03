from abc import ABC, abstractmethod
from typing import List, Optional

from supercontrast.optimizer import Optimizer
from supercontrast.provider import Provider, ProviderHandler


class TaskHandler(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def request(self, *args, **kwargs):
        raise NotImplementedError("TaskHandler must implement request method")

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("TaskHandler must implement evaluate method")

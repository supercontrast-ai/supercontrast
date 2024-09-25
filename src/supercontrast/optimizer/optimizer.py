from abc import ABC, abstractmethod
from typing import List

from supercontrast.providers.provider import Provider, ProviderType


class OptimizerFunction(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Optimizer(ABC):
    def __init__(self, optimizer_function: OptimizerFunction, providers: List[ProviderType]):
        self.optimizer_function = optimizer_function
        self.providers = providers

    def watch(self):
        pass

    def __call__(self) -> ProviderType:
        raise NotImplementedError("Subclass must implement abstract method")
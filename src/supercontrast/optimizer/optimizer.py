from abc import ABC, abstractmethod
from typing import List

from supercontrast.providers import Provider, ProviderModel


class OptimizerFunction(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Optimizer(ABC):
    def __init__(
        self, optimizer_function: OptimizerFunction, providers: List[Provider]
    ):
        self.optimizer_function = optimizer_function
        self.providers = providers

    def watch(self):
        pass

    def __call__(self) -> ProviderModel:
        raise NotImplementedError("Subclass must implement abstract method")

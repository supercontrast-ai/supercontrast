from abc import ABC
from typing import List, Optional

from supercontrast.optimizer.optimizer import Optimizer, OptimizerFunction
from supercontrast.provider import Provider, ProviderModel


class TaskHandler(ABC):
    def __init__(
        self,
        providers: List[Provider],
        optimize_by: Optional[OptimizerFunction] = None,
    ):
        if len(providers) > 1 and optimize_by == None:
            raise ValueError(
                "Optimizer function must be provided if more than one provider is used"
            )

        if optimize_by != None:
            self.optimizer = Optimizer(optimize_by, providers)
            self.optimizer.watch()
        else:
            self.optimizer = None

        self.provider_connections = {}

    def _get_provider(self) -> ProviderModel:
        if self.optimizer:
            provider_value = self.optimizer()
            connection = self.provider_connections.get(provider_value)

            if connection is None:
                raise ValueError(f"Provider {provider_value} not found")

            return connection
        else:
            providers = list(self.provider_connections.values())
            if len(providers) == 0:
                raise ValueError("No provider connections found")

            return providers[0]

    def request(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

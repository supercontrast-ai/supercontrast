from supercontrast.optimizer.optimizer_handler import OptimizerHandler
from supercontrast.provider.provider_enum import Provider


class OptimizerMockHandler(OptimizerHandler):
    def get_provider(self) -> Provider:
        if not self.providers:
            raise ValueError("No providers available")
        return self.providers[0]

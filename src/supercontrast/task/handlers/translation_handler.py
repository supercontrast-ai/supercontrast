from pydantic import BaseModel
from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.provider import Provider
from supercontrast.provider.provider_factory import provider_factory
from supercontrast.task.task_enum import Task
from supercontrast.task.task_handler import TaskHandler
from supercontrast.task.types.translation_types import (
    TranslationRequest,
    TranslationResponse,
)


class TranslationHandler(TaskHandler):
    def __init__(
        self,
        providers: List[Provider],
        optimize_by: Optional[OptimizerFunction] = None,
        **config
    ):
        super().__init__(providers, optimize_by)
        self.provider_connections = {
            provider: provider_factory(
                task=Task.TRANSLATION, provider=provider, **config
            )
            for provider in providers
        }

    def request(self, body: TranslationRequest) -> TranslationResponse:
        provider = self._get_provider()
        return provider.request(body)

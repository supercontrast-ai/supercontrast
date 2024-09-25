from typing import List, Optional, Union

from pydantic import BaseModel
from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers.provider import ProviderType, provider_factory
from supercontrast.tasks.task_handler import TaskHandler
from supercontrast.tasks.task_types import Task

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    text: str

class TranslationHandler(TaskHandler):
    def __init__(self, providers: List[ProviderType], optimize_by: Optional[OptimizerFunction] = None, **config):
        super().__init__(providers, optimize_by)
        self.provider_connections = {
            provider: provider_factory(task=Task.TRANSLATION, provider=provider, **config)
            for provider in providers
        }

    def request(self, body: TranslationRequest) -> TranslationResponse:
        provider = self._get_provider()
        return provider.request(body)
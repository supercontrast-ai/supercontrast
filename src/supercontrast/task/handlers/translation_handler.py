from typing import List, Optional

from supercontrast.optimizer import Optimizer
from supercontrast.optimizer.optimizer_factory import optimizer_factory
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
        self, providers: List[Provider], optimizer: Optional[Optimizer] = None, **config
    ):
        self.task = Task.TRANSLATION
        self.provider_handler_map = {
            provider: provider_factory(task=Task.TRANSLATION, provider=provider)
            for provider in providers
        }
        self.optimizer_handler = optimizer_factory(
            task=Task.TRANSLATION, providers=providers, optimizer=optimizer
        )

    def request(self, body: TranslationRequest) -> TranslationResponse:
        provider = self.optimizer_handler.get_provider()
        provider_handler = self.provider_handler_map[provider]
        return provider_handler.request(body)

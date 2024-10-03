from typing import Dict, List, Optional

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
            provider: provider_factory(
                task=Task.TRANSLATION, provider=provider, **config
            )
            for provider in providers
        }
        self.optimizer_handler = optimizer_factory(
            task=Task.TRANSLATION, providers=providers, optimizer=optimizer
        )

    def request(
        self, body: TranslationRequest, provider: Optional[Provider] = None
    ) -> TranslationResponse:
        if provider is None:
            provider = self.optimizer_handler.get_provider()

        provider_handler = self.provider_handler_map[provider]
        return provider_handler.request(body)

    def evaluate(self, body: TranslationRequest) -> Dict[Provider, TranslationResponse]:
        responses = {}
        for provider, handler in self.provider_handler_map.items():
            try:
                response = handler.request(body)
                responses[provider] = response
            except Exception as e:
                # Log the error or handle it as appropriate for your use case
                print(f"Error evaluating provider {provider}: {str(e)}")
        return responses

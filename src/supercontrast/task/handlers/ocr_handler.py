from typing import List, Optional

from supercontrast.optimizer import Optimizer
from supercontrast.optimizer.optimizer_factory import optimizer_factory
from supercontrast.provider import Provider
from supercontrast.provider.provider_factory import provider_factory
from supercontrast.task.task_enum import Task
from supercontrast.task.task_handler import TaskHandler
from supercontrast.task.types.ocr_types import OCRRequest, OCRResponse


class OCRHandler(TaskHandler):
    def __init__(
        self,
        providers: List[Provider],
        optimizer: Optional[Optimizer] = None,
    ):
        self.task = Task.OCR
        self.provider_handler_map = {
            provider: provider_factory(task=Task.OCR, provider=provider)
            for provider in providers
        }
        self.optimizer_handler = optimizer_factory(
            task=Task.OCR, providers=providers, optimizer=optimizer
        )

    def request(self, body: OCRRequest) -> OCRResponse:
        provider = self.optimizer_handler.get_provider()
        provider_handler = self.provider_handler_map[provider]
        return provider_handler.request(body)

from pydantic import BaseModel
from typing import List, Optional, Union

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers import Provider, provider_factory
from supercontrast.tasks.task_enum import Task
from supercontrast.tasks.task_handler import TaskHandler
from supercontrast.tasks.types.ocr_types import OCRRequest, OCRResponse


class OCRHandler(TaskHandler):
    def __init__(
        self,
        providers: List[Provider],
        optimize_by: Optional[OptimizerFunction] = None,
    ):
        super().__init__(providers, optimize_by)
        self.provider_connections = {
            provider: provider_factory(task=Task.OCR, provider=provider)
            for provider in providers
        }

    def request(self, body: OCRRequest) -> OCRResponse:
        provider = self._get_provider()
        return provider.request(body)

from typing import List, Optional, Union

from pydantic import BaseModel
from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers.provider import ProviderType, provider_factory
from supercontrast.tasks.task_handler import TaskHandler
from supercontrast.tasks.task_types import Task

class OCRRequest(BaseModel):
    image: Union[str, bytes]

class OCRResponse(BaseModel):
    text: str

class OCRHandler(TaskHandler):
    def __init__(self, providers: List[ProviderType], optimize_by: Optional[OptimizerFunction] = None):
        super().__init__(providers, optimize_by)
        self.provider_connections = {
            provider: provider_factory(task=Task.OCR, provider=provider)
            for provider in providers
        }

    def request(self, body: OCRRequest) -> OCRResponse:
        provider = self._get_provider()
        return provider.request(body)
from pydantic import BaseModel
from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers.provider import ProviderType, provider_factory
from supercontrast.tasks.task_handler import TaskHandler
from supercontrast.tasks.task_types import Task


class SentimentAnalysisRequest(BaseModel):
    text: str


class SentimentAnalysisResponse(BaseModel):
    score: float


class SentimentAnalysisHandler(TaskHandler):
    def __init__(
        self,
        providers: List[ProviderType],
        optimize_by: Optional[OptimizerFunction] = None,
    ):
        super().__init__(providers, optimize_by)
        self.provider_connections = {
            provider: provider_factory(task=Task.SENTIMENT_ANALYSIS, provider=provider)
            for provider in providers
        }

    def request(self, body: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        provider = self._get_provider()
        return provider.request(body)

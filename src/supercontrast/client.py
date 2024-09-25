from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers.provider import Provider, ProviderType
from supercontrast.tasks.ocr import OCRHandler
from supercontrast.tasks.sentiment_analysis import SentimentAnalysisHandler
from supercontrast.tasks.task_types import Task
from supercontrast.tasks.translation import TranslationHandler

HANDLER_MAP = {
    Task.SENTIMENT_ANALYSIS: SentimentAnalysisHandler,
    Task.TRANSLATION: TranslationHandler,
    Task.OCR: OCRHandler,
}


def supercontrast_client(
    task: Task,
    providers: List[ProviderType],
    optimize_by: Optional[OptimizerFunction] = None,
    **config
):
    handler_cls = HANDLER_MAP[task]
    return handler_cls(providers, optimize_by, **config)

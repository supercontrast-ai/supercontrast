from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.providers import Provider
from supercontrast.tasks import Task
from supercontrast.tasks.handlers import (
    OCRHandler,
    SentimentAnalysisHandler,
    TranslationHandler,
)

HANDLER_MAP = {
    Task.SENTIMENT_ANALYSIS: SentimentAnalysisHandler,
    Task.TRANSLATION: TranslationHandler,
    Task.OCR: OCRHandler,
}


def supercontrast_client(
    task: Task,
    providers: List[Provider],
    optimize_by: Optional[OptimizerFunction] = None,
    **config
):
    handler_cls = HANDLER_MAP[task]
    return handler_cls(providers, optimize_by, **config)

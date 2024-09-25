from typing import List, Optional

from supercontrast.optimizer.optimizer import OptimizerFunction
from supercontrast.provider import Provider
from supercontrast.task.handlers import (
    OCRHandler,
    SentimentAnalysisHandler,
    TranslationHandler,
)
from supercontrast.task.task_enum import Task


def task_factory(
    task: Task,
    providers: List[Provider],
    optimize_by: Optional[OptimizerFunction] = None,
    **config,
):
    if task == Task.SENTIMENT_ANALYSIS:
        return SentimentAnalysisHandler(providers, optimize_by, **config)
    elif task == Task.TRANSLATION:
        return TranslationHandler(providers, optimize_by, **config)
    elif task == Task.OCR:
        return OCRHandler(providers, optimize_by, **config)
    else:
        raise ValueError(f"Unsupported task: {task}")

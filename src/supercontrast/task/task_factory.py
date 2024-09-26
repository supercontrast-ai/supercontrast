from typing import List, Optional

from supercontrast.optimizer import Optimizer
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
    optimizer: Optional[Optimizer] = None,
    **config,
):
    if task == Task.SENTIMENT_ANALYSIS:
        return SentimentAnalysisHandler(providers, optimizer, **config)
    elif task == Task.TRANSLATION:
        return TranslationHandler(providers, optimizer, **config)
    elif task == Task.OCR:
        return OCRHandler(providers, optimizer, **config)
    else:
        raise ValueError(f"Unsupported task: {task}")

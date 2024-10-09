from typing import Optional

from supercontrast.metrics.metrics_handler import (
    MetricsHandler,
    OCRMetricsHandler,
    TranscriptionMetricsHandler,
    TranslationMetricsHandler,
)
from supercontrast.task.task_enum import Task


# metrics factory
def metrics_factory(task: Task) -> Optional[MetricsHandler]:
    if task == Task.OCR:
        return OCRMetricsHandler()
    elif task == Task.TRANSCRIPTION:
        return TranscriptionMetricsHandler()
    elif task == Task.TRANSLATION:
        return TranslationMetricsHandler()
    elif task == Task.SENTIMENT_ANALYSIS:
        return None
    else:
        raise ValueError(f"Unsupported task: {task}")

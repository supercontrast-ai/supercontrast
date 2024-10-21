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
    else:
        return None

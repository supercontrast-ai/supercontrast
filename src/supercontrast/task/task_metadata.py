from pydantic import BaseModel
from typing import Any, Dict, Generic, Optional, TypeVar

from supercontrast.metrics.metrics_enum import Metric
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task

ResponseType = TypeVar("ResponseType")


class TaskMetadata(BaseModel, Generic[ResponseType]):
    task: Task
    provider: Provider
    latency: float
    reference: Optional[ResponseType] = None
    normalized_reference: Optional[ResponseType] = None
    normalized_prediction: Optional[ResponseType] = None
    metrics: Optional[Dict[Metric, Any]] = None

    def __str__(self):
        lines = [
            f"Task: {self.task}",
            f"Provider: {self.provider}",
            f"Latency: {self.latency:.2f}s",
        ]
        if self.reference:
            lines.append(f"Reference: {self.reference}")
        if self.normalized_reference:
            lines.append(f"Normalized Reference: {self.normalized_reference}")
        if self.normalized_prediction:
            lines.append(f"Normalized Prediction: {self.normalized_prediction}")
        if self.metrics:
            lines.append("Metrics:")
            for metric, value in self.metrics.items():
                lines.append(f"  {metric}: {value}")
        return "\n".join(lines)

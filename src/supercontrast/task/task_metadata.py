from pydantic import BaseModel
from typing import Any, Dict

from supercontrast.metrics.metrics_enum import Metric
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task


class TaskMetadata(BaseModel):
    task: Task
    provider: Provider
    latency: float
    metrics: Dict[Metric, Any]

from pydantic import BaseModel
from typing import Any, Dict, Generic, TypeVar

from supercontrast.metrics.metrics_enum import Metric

ResponseType = TypeVar("ResponseType")


class MetricsResponse(BaseModel, Generic[ResponseType]):
    metrics: Dict[Metric, Any]
    normalized_reference: ResponseType
    normalized_prediction: ResponseType

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return (
            f"MetricsResponse(metrics={self.metrics}, "
            f"normalized_reference={self.normalized_reference}, "
            f"normalized_prediction={self.normalized_prediction})"
        )

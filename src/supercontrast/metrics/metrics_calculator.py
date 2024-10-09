from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from supercontrast.metrics.metrics_enum import Metric

# metrics calculator


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class MetricsCalculator(ABC, Generic[InputType, OutputType]):
    @abstractmethod
    def __init__(self, metric: Metric, *args, **kwargs):
        self.metric = metric

    @abstractmethod
    def calculate(self, reference: InputType, hypothesis: InputType) -> OutputType:
        raise NotImplementedError("MetricsCalculator must implement calculate method")

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("MetricsCalculator must implement get_name method")

from jiwer import cer

from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric


class CharacterErrorRateCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.CER, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        result = cer(reference, hypothesis)
        if isinstance(result, float):
            return result
        else:
            raise ValueError(f"Invalid result type: {type(result)}")

    def get_name(self) -> str:
        return "Character Error Rate"

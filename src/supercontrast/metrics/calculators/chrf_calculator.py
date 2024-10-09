from nltk.translate.chrf_score import sentence_chrf

from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric


class CHRFCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.CHRF, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return sentence_chrf(hypothesis, [reference])

    def get_name(self) -> str:
        return "chrF Score"

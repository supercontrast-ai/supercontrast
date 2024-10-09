from jiwer import mer, wer, wil

from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric


class WordErrorRateCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.WER, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return wer(reference, hypothesis)

    def get_name(self) -> str:
        return "Word Error Rate"


class MatchErrorRateCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.MER, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return mer(reference, hypothesis)

    def get_name(self) -> str:
        return "Match Error Rate"


class WordInformationLostCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.WIL, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return wil(reference, hypothesis)

    def get_name(self) -> str:
        return "Word Information Lost"


class WordInformationPreservedCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.WIP, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return 1 - wil(reference, hypothesis)

    def get_name(self) -> str:
        return "Word Information Preserved"


class WordRecognitionRateCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.WRR, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return 1 - wer(reference, hypothesis)

    def get_name(self) -> str:
        return "Word Recognition Rate"

import nltk

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score

from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric


class METEORCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.METEOR, *args, **kwargs)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference_tokens = word_tokenize(reference)
        hypothesis_tokens = word_tokenize(hypothesis)
        return single_meteor_score(reference_tokens, hypothesis_tokens)

    def get_name(self) -> str:
        return "METEOR Score"

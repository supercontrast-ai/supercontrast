import nltk

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
from sacrebleu import sentence_bleu

from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric


class BLEUCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.BLEU, *args, **kwargs)

    def calculate(self, reference: str, hypothesis: str) -> float:
        return sentence_bleu(hypothesis, [reference]).score

    def get_name(self) -> str:
        return "BLEU Score"


class BLEUNLTKCalculator(MetricsCalculator[str, float]):
    def __init__(self, *args, **kwargs):
        super().__init__(Metric.BLEU_NLTK, *args, **kwargs)
        nltk.download("punkt", quiet=True)

    def calculate(self, reference: str, hypothesis: str) -> float:
        reference_tokens = word_tokenize(reference)
        hypothesis_tokens = word_tokenize(hypothesis)
        bleu_score = nltk_sentence_bleu([reference_tokens], hypothesis_tokens)
        if isinstance(bleu_score, float):
            return bleu_score
        if isinstance(bleu_score, int):
            return float(bleu_score)
        else:
            raise ValueError(f"Invalid result type: {type(bleu_score)}")

    def get_name(self) -> str:
        return "BLEU Score (NLTK)"

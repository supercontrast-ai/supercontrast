from abc import ABC, abstractmethod
from typing import Any, Dict

from supercontrast.metrics.calculators.bleu_calculator import (
    BLEUCalculator,
    BLEUNLTKCalculator,
)
from supercontrast.metrics.calculators.character_calculator import (
    CharacterErrorRateCalculator,
)
from supercontrast.metrics.calculators.chrf_calculator import CHRFCalculator
from supercontrast.metrics.calculators.meteor_calculator import METEORCalculator
from supercontrast.metrics.calculators.word_calculator import (
    MatchErrorRateCalculator,
    WordErrorRateCalculator,
    WordInformationLostCalculator,
    WordInformationPreservedCalculator,
    WordRecognitionRateCalculator,
)
from supercontrast.metrics.metrics_calculator import MetricsCalculator
from supercontrast.metrics.metrics_enum import Metric
from supercontrast.task.types.ocr_types import OCRResponse
from supercontrast.task.types.transcription_types import TranscriptionResponse
from supercontrast.task.types.translation_types import TranslationResponse


def get_metrics_calculator(metric: Metric, **config) -> MetricsCalculator:
    if metric == Metric.BLEU:
        return BLEUCalculator(**config)
    elif metric == Metric.BLEU_NLTK:
        return BLEUNLTKCalculator(**config)
    elif metric == Metric.CER:
        return CharacterErrorRateCalculator(**config)
    elif metric == Metric.WER:
        return WordErrorRateCalculator(**config)
    elif metric == Metric.MER:
        return MatchErrorRateCalculator(**config)
    elif metric == Metric.WIL:
        return WordInformationLostCalculator(**config)
    elif metric == Metric.WIP:
        return WordInformationPreservedCalculator(**config)
    elif metric == Metric.WRR:
        return WordRecognitionRateCalculator(**config)
    elif metric == Metric.METEOR:
        return METEORCalculator(**config)
    elif metric == Metric.CHRF:
        return CHRFCalculator(**config)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class MetricsHandler(ABC):
    def __init__(self):
        self.metrics_calculators = self.get_metrics_calculators()

    @abstractmethod
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        pass

    @abstractmethod
    def calculate_metrics(self, reference: Any, prediction: Any) -> Dict[Metric, float]:
        pass


class OCRMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.CER: get_metrics_calculator(Metric.CER),
            Metric.WER: get_metrics_calculator(Metric.WER),
        }

    def calculate_metrics(
        self, reference: OCRResponse, prediction: OCRResponse
    ) -> Dict[Metric, float]:
        results = {}
        for metric, calculator in self.metrics_calculators.items():
            results[metric] = calculator.calculate(
                reference.all_text, prediction.all_text
            )
        return results


class TranscriptionMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.WER: get_metrics_calculator(Metric.WER),
            Metric.MER: get_metrics_calculator(Metric.MER),
            Metric.WIL: get_metrics_calculator(Metric.WIL),
            Metric.CER: get_metrics_calculator(Metric.CER),
            Metric.WIP: get_metrics_calculator(Metric.WIP),
            Metric.WRR: get_metrics_calculator(Metric.WRR),
        }

    def calculate_metrics(
        self, reference: TranscriptionResponse, prediction: TranscriptionResponse
    ) -> Dict[Metric, float]:
        results = {}
        for metric, calculator in self.metrics_calculators.items():
            results[metric] = calculator.calculate(reference.text, prediction.text)
        return results


class TranslationMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.BLEU: get_metrics_calculator(Metric.BLEU),
            Metric.BLEU_NLTK: get_metrics_calculator(Metric.BLEU_NLTK),
            Metric.METEOR: get_metrics_calculator(Metric.METEOR),
            Metric.CHRF: get_metrics_calculator(Metric.CHRF),
        }

    def calculate_metrics(
        self, reference: TranslationResponse, prediction: TranslationResponse
    ) -> Dict[Metric, float]:
        results = {}
        for metric, calculator in self.metrics_calculators.items():
            results[metric] = calculator.calculate(reference.text, prediction.text)
        return results

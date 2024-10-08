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
from supercontrast.metrics.metrics_response import MetricsResponse
from supercontrast.task.types.ocr_types import OCRResponse
from supercontrast.task.types.transcription_types import TranscriptionResponse
from supercontrast.task.types.translation_types import TranslationResponse
from supercontrast.utils.text import normalize_text

# get metrics calculator


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


# metrics handler


class MetricsHandler(ABC):
    def __init__(self):
        self.metrics_calculators = self.get_metrics_calculators()

    @abstractmethod
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        pass

    @abstractmethod
    def calculate_metrics(self, reference: Any, prediction: Any) -> MetricsResponse:
        pass


class OCRMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.CER: CharacterErrorRateCalculator(),
            Metric.WER: WordErrorRateCalculator(),
        }

    def calculate_metrics(
        self, reference: OCRResponse, prediction: OCRResponse
    ) -> MetricsResponse:
        normalized_reference = normalize_text(reference.all_text, "ocr")
        normalized_prediction = normalize_text(prediction.all_text, "ocr")

        metrics: Dict[Metric, Any] = {}
        for metric, calculator in self.metrics_calculators.items():
            metrics[metric] = calculator.calculate(
                normalized_reference, normalized_prediction
            )
        return MetricsResponse(
            metrics=metrics,
            normalized_reference=normalized_reference,
            normalized_prediction=normalized_prediction,
        )


class TranscriptionMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.WER: WordErrorRateCalculator(),
            Metric.MER: MatchErrorRateCalculator(),
            Metric.WIL: WordInformationLostCalculator(),
            Metric.CER: CharacterErrorRateCalculator(),
            Metric.WIP: WordInformationPreservedCalculator(),
            Metric.WRR: WordRecognitionRateCalculator(),
        }

    def calculate_metrics(
        self, reference: TranscriptionResponse, prediction: TranscriptionResponse
    ) -> MetricsResponse:
        normalized_reference = normalize_text(reference.text, "transcription")
        normalized_prediction = normalize_text(prediction.text, "transcription")

        metrics = {}
        for metric, calculator in self.metrics_calculators.items():
            metrics[metric] = calculator.calculate(
                normalized_reference, normalized_prediction
            )
        return MetricsResponse(
            metrics=metrics,
            normalized_reference=normalized_reference,
            normalized_prediction=normalized_prediction,
        )


class TranslationMetricsHandler(MetricsHandler):
    def get_metrics_calculators(self) -> Dict[Metric, MetricsCalculator]:
        return {
            Metric.BLEU: BLEUCalculator(),
            Metric.BLEU_NLTK: BLEUNLTKCalculator(),
            Metric.METEOR: METEORCalculator(),
            Metric.CHRF: CHRFCalculator(),
        }

    def calculate_metrics(
        self, reference: TranslationResponse, prediction: TranslationResponse
    ) -> MetricsResponse:
        normalized_reference = normalize_text(reference.text, "translation")
        normalized_prediction = normalize_text(prediction.text, "translation")

        metrics = {}
        for metric, calculator in self.metrics_calculators.items():
            metrics[metric] = calculator.calculate(
                normalized_reference, normalized_prediction
            )
        return MetricsResponse(
            metrics=metrics,
            normalized_reference=normalized_reference,
            normalized_prediction=normalized_prediction,
        )

import io
import os
import time

from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.translation.text import TextTranslationClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.core.credentials import AzureKeyCredential
from msrest.authentication import CognitiveServicesCredentials

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task import (
    OCRRequest,
    OCRResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    Task,
    TranslationRequest,
    TranslationResponse,
)

# models


class AzureSentimentAnalysis(ProviderHandler):
    def __init__(self, endpoint: str, key: str):
        super().__init__(provider=Provider.AZURE, task=Task.SENTIMENT_ANALYSIS)
        self.client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        response = self.client.analyze_sentiment([request.text])[0]
        score = (
            response.confidence_scores.positive - response.confidence_scores.negative
        )
        return SentimentAnalysisResponse(score=score)

    def get_name(self) -> str:
        return "Azure Text Analytics - Sentiment Analysis"

    @classmethod
    def init_from_env(cls, endpoint=None, key=None) -> "AzureSentimentAnalysis":
        endpoint = endpoint or os.environ.get("AZURE_TEXT_ANALYTICS_ENDPOINT")
        key = key or os.environ.get("AZURE_TEXT_ANALYTICS_KEY")
        if not endpoint or not key:
            raise ValueError(
                "AZURE_TEXT_ANALYTICS_ENDPOINT and AZURE_TEXT_ANALYTICS_KEY must be set"
            )

        return cls(endpoint, key)


class AzureTranslation(ProviderHandler):
    def __init__(
        self, key: str, region: str, source_language: str, target_language: str
    ):
        super().__init__(provider=Provider.AZURE, task=Task.TRANSLATION)
        self.client = TextTranslationClient(
            credential=AzureKeyCredential(key), region=region
        )
        self.source_language = source_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        response = self.client.translate(
            body=[request.text],
            from_language=self.source_language,
            to_language=[self.target_language],
        )
        translated_text = response[0].translations[0].text
        return TranslationResponse(text=translated_text)

    def get_name(self) -> str:
        return "Azure Translator"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str, key=None, region=None
    ) -> "AzureTranslation":
        key = key or os.environ.get("AZURE_TEXT_ANALYTICS_KEY")
        region = region or os.environ.get("AZURE_TRANSLATOR_REGION")
        if not key or not region:
            raise ValueError(
                "AZURE_TEXT_ANALYTICS_KEY and AZURE_TRANSLATOR_REGION must be set"
            )

        return cls(key, region, source_language, target_language)


class AzureOCR(ProviderHandler):
    def __init__(self, endpoint: str, key: str):
        super().__init__(provider=Provider.AZURE, task=Task.OCR)
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            read_response = self.client.read(request.image, raw=True)
        else:
            read_response = self.client.read_in_stream(
                io.BytesIO(request.image), raw=True
            )

        if not read_response:
            raise ValueError("Failed to read image")

        operation_location = read_response.headers["Operation-Location"]

        if not operation_location:
            raise ValueError("Failed to get operation location")

        operation_id = operation_location.split("/")[-1]

        while True:
            read_result = self.client.get_read_result(operation_id)

            if read_result.status not in ["notStarted", "running"]:  # type: ignore
                break
            time.sleep(1)

        extracted_text = ""

        if read_result.status == OperationStatusCodes.succeeded:  # type: ignore
            for text_result in read_result.analyze_result.read_results:  # type: ignore
                for line in text_result.lines:
                    extracted_text += line.text + "\n"

        return OCRResponse(text=extracted_text.strip())

    def get_name(self) -> str:
        return "Azure Computer Vision - OCR"

    @classmethod
    def init_from_env(cls, endpoint=None, key=None) -> "AzureOCR":
        endpoint = endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
        key = key or os.environ.get("AZURE_VISION_KEY")
        if not endpoint or not key:
            raise ValueError("AZURE_VISION_ENDPOINT and AZURE_VISION_KEY must be set")

        return cls(endpoint, key)


# factory


def azure_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.SENTIMENT_ANALYSIS:
        endpoint = config.get("azure_text_analytics_endpoint")
        key = config.get("azure_text_analytics_key")
        return AzureSentimentAnalysis.init_from_env(endpoint=endpoint, key=key)
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        key = config.get("azure_text_analytics_key")
        region = config.get("azure_translator_region")
        return AzureTranslation.init_from_env(
            source_language=source_language, target_language=target_language,
            key=key, region=region
        )
    elif task == Task.OCR:
        endpoint = config.get("azure_vision_endpoint")
        key = config.get("azure_vision_key")
        return AzureOCR.init_from_env(endpoint=endpoint, key=key)
    else:
        raise ValueError(f"Unsupported task: {task}")

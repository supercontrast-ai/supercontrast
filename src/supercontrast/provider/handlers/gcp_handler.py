import os

from google.cloud import language_v1, translate_v2, vision_v1
from google.oauth2 import service_account

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


class GCPSentimentAnalysis(ProviderHandler):
    def __init__(self, credentials):
        super().__init__(provider=Provider.GCP, task=Task.SENTIMENT_ANALYSIS)
        self.client = language_v1.LanguageServiceClient(credentials=credentials)
        self.THRESHOLD = 0

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        document = language_v1.Document(
            content=request.text, type_=language_v1.Document.Type.PLAIN_TEXT
        )
        sentiment = self.client.analyze_sentiment(
            request={"document": document}
        ).document_sentiment

        score = sentiment.score

        return SentimentAnalysisResponse(score=score)

    def get_name(self) -> str:
        return "Google Natural Language - Sentiment Analysis"

    @classmethod
    def init_from_env(cls) -> "GCPSentimentAnalysis":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return cls(credentials)


class GCPTranslation(ProviderHandler):
    def __init__(self, credentials, src_language: str, target_language: str):
        super().__init__(provider=Provider.GCP, task=Task.TRANSLATION)
        self.client = translate_v2.Client(credentials=credentials)
        self.src_language = src_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        result = self.client.translate(
            request.text,
            source_language=self.src_language,
            target_language=self.target_language,
        )
        translated_text = result["translatedText"]

        return TranslationResponse(text=translated_text)

    def get_name(self) -> str:
        return "Google Translation"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "GCPTranslation":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return cls(credentials, source_language, target_language)


class GCPOCR(ProviderHandler):
    def __init__(self, credentials):
        super().__init__(provider=Provider.GCP, task=Task.OCR)
        self.client = vision_v1.ImageAnnotatorClient(credentials=credentials)

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            if request.image.startswith(("http://", "https://")):
                image = vision_v1.Image(
                    source=vision_v1.ImageSource(image_uri=request.image)
                )
            else:
                with open(request.image, "rb") as image_file:
                    content = image_file.read()
                image = vision_v1.Image(content=content)
        else:
            image = vision_v1.Image(content=request.image)

        response = self.client.document_text_detection(image=image)  # type: ignore

        extracted_text = response.full_text_annotation.text  # type: ignore

        return OCRResponse(text=extracted_text.strip())

    def get_name(self) -> str:
        return "Google Vision - OCR"

    @classmethod
    def init_from_env(cls) -> "GCPOCR":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return cls(credentials)


# factory


def gcp_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.SENTIMENT_ANALYSIS:
        return GCPSentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return GCPTranslation.init_from_env(
            source_language=source_language,
            target_language=target_language,
        )
    elif task == Task.OCR:
        return GCPOCR.init_from_env()
    else:
        raise ValueError(f"Unsupported task: {task}")

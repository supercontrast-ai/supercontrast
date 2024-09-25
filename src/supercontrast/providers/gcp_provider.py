import os

from google.cloud import language_v1, translate_v2, vision_v1

from supercontrast.providers.provider import Provider
from supercontrast.tasks.ocr import OCRRequest, OCRResponse
from supercontrast.tasks.sentiment_analysis import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from supercontrast.tasks.translation import TranslationRequest, TranslationResponse


class GCPSentimentAnalysis(Provider):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = language_v1.LanguageServiceClient(
            client_options={"api_key": api_key}
        )
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
        api_key = os.environ.get("GCP_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY environment variable not set"
            )
        return cls(api_key)


class GCPTranslation(Provider):
    def __init__(self, api_key: str, src_language: str, target_language: str):
        super().__init__()
        self.client = translate_v2.Client(client_options={"api_key": api_key})
        self.src_language = src_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        result = self.client.translate(
            request.text,
        )
        translated_text = result["translatedText"]

        return TranslationResponse(text=translated_text)

    def get_name(self) -> str:
        return "Google Translation"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "GCPTranslation":
        api_key = os.environ.get("GCP_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY environment variable not set"
            )
        return cls(api_key, source_language, target_language)


class GCPOCR(Provider):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = vision_v1.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            with open(request.image, "rb") as image_file:
                content = image_file.read()
        else:
            content = request.image

        image = vision_v1.Image(content=content)
        response = self.client.document_text_detection(image=image)

        extracted_text = response.full_text_annotation.text

        return OCRResponse(text=extracted_text.strip())

    def get_name(self) -> str:
        return "Google Vision - OCR"

    @classmethod
    def init_from_env(cls) -> "GCPOCR":
        api_key = os.environ.get("GCP_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY environment variable not set"
            )
        return cls(api_key)

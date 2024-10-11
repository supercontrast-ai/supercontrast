import os

from google.cloud import language_v1, translate_v2, vision_v1
from google.oauth2 import service_account

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRBoundingBox, OCRRequest, OCRResponse
from supercontrast.task.types.sentiment_analysis_types import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from supercontrast.task.types.translation_types import (
    TranslationRequest,
    TranslationResponse,
)
from supercontrast.utils.image import load_image_data

# Constants

GCP_SUPPORTED_TASKS = [Task.SENTIMENT_ANALYSIS, Task.TRANSLATION, Task.OCR]

# Task.SENTIMENT_ANALYSIS


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
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set"
            )
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return cls(credentials)


# Task.TRANSLATION


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
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set"
            )
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return cls(credentials, source_language, target_language)


# Task.OCR


class GCPOCR(ProviderHandler):
    def __init__(self, credentials):
        super().__init__(provider=Provider.GCP, task=Task.OCR)
        self.client = vision_v1.ImageAnnotatorClient(credentials=credentials)

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)
        image = vision_v1.Image(content=image_data)

        response = self.client.document_text_detection(image=image)  # type: ignore

        extracted_text = response.full_text_annotation.text  # type: ignore

        bounding_boxes = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        vertices = [
                            (vertex.x, vertex.y)
                            for vertex in word.bounding_box.vertices
                        ]
                        bounding_boxes.append(
                            OCRBoundingBox(text=word_text, coordinates=vertices)
                        )

        response = OCRResponse(
            all_text=extracted_text.strip(), bounding_boxes=bounding_boxes
        )
        return response

    def get_name(self) -> str:
        return "Google Vision - OCR"

    @classmethod
    def init_from_env(cls) -> "GCPOCR":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set"
            )
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return cls(credentials)


# factory


def gcp_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in GCP_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.SENTIMENT_ANALYSIS:
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

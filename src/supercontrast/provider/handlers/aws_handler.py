import boto3
import requests
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
from supercontrast.utils.text import truncate_text

# models


class AWSSentimentAnalysis(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.AWS, task=Task.SENTIMENT_ANALYSIS)
        self.client = boto3.client("comprehend")
        self.THRESHOLD = 0

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        response = self.client.detect_sentiment(
            Text=truncate_text(request.text), LanguageCode="en"
        )
        score = (
            response["SentimentScore"]["Positive"]
            - response["SentimentScore"]["Negative"]
        )

        return SentimentAnalysisResponse(score=score)

    def get_name(self) -> str:
        return "Aws Comprehend - Sentiment Analysis"

    @classmethod
    def init_from_env(cls) -> "AWSSentimentAnalysis":
        return cls()


class AWSTranslate(ProviderHandler):
    def __init__(self, src_language: str, target_language: str):
        super().__init__(provider=Provider.AWS, task=Task.TRANSLATION)
        self.client = boto3.client("translate")
        self.src_language = src_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        response = self.client.translate_text(
            Text=truncate_text(request.text),
            SourceLanguageCode=self.src_language,
            TargetLanguageCode=self.target_language,
        )
        translated_text = response["TranslatedText"]

        result = TranslationResponse(
            text=translated_text,
        )

        return result

    def get_name(self) -> str:
        return "AWS Translate"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "AWSTranslate":
        return cls(source_language, target_language)


class AWSOCR(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.AWS, task=Task.OCR)
        self.client = boto3.client("textract")

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            if request.image.startswith(('http://', 'https://')):
                image_data = requests.get(request.image).content
            else:
                with open(request.image, 'rb') as image_file:
                    image_data = image_file.read()
        else:
            image_data = request.image

        response = self.client.analyze_document(
            Document={"Bytes": image_data}, FeatureTypes=["FORMS", "TABLES"]
        )

        extracted_text = ""
        for item in response.get("Blocks", []):
            if item["BlockType"] == "LINE":
                extracted_text += item["Text"] + "\n"

        return OCRResponse(text=extracted_text.strip())

    def get_name(self) -> str:
        return "AWS Textract - OCR"

    @classmethod
    def init_from_env(cls) -> "AWSOCR":
        return cls()


# factory


def aws_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.SENTIMENT_ANALYSIS:
        return AWSSentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return AWSTranslate.init_from_env(
            source_language=source_language, target_language=target_language
        )
    elif task == Task.OCR:
        return AWSOCR.init_from_env()
    else:
        raise ValueError(f"Unsupported task: {task}")

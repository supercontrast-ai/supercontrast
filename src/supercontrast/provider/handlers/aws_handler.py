import boto3

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
from supercontrast.utils.image import get_image_size, load_image_data
from supercontrast.utils.text import truncate_text

# Constants

AWS_SUPPORTED_TASKS = [Task.SENTIMENT_ANALYSIS, Task.TRANSLATION, Task.OCR]

# Task.SENTIMENT_ANALYSIS


class AWSSentimentAnalysis(ProviderHandler):
    def __init__(
        self, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None
    ):
        super().__init__(provider=Provider.AWS, task=Task.SENTIMENT_ANALYSIS)
        self.client = boto3.client(
            "comprehend",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
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
    def init_from_env(
        cls, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None
    ) -> "AWSSentimentAnalysis":
        return cls(aws_access_key_id, aws_secret_access_key, aws_session_token)


# Task.TRANSLATION


class AWSTranslate(ProviderHandler):
    def __init__(
        self,
        src_language: str,
        target_language: str,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
    ):
        super().__init__(provider=Provider.AWS, task=Task.TRANSLATION)
        self.client = boto3.client(
            "translate",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
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
        cls,
        source_language: str,
        target_language: str,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
    ) -> "AWSTranslate":
        return cls(
            source_language,
            target_language,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
        )


# Task.OCR


class AWSOCR(ProviderHandler):
    def __init__(
        self, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None
    ):
        super().__init__(provider=Provider.AWS, task=Task.OCR)
        self.client = boto3.client(
            "textract",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)

        # Get image dimensions
        width, height = get_image_size(request.image)

        response = self.client.analyze_document(
            Document={"Bytes": image_data}, FeatureTypes=["FORMS", "TABLES"]
        )

        extracted_text = ""
        bounding_boxes = []
        for item in response.get("Blocks", []):
            if item["BlockType"] == "LINE":
                extracted_text += item["Text"] + "\n"
                if "Geometry" in item and "BoundingBox" in item["Geometry"]:
                    bbox = item["Geometry"]["BoundingBox"]
                    left = int(bbox["Left"] * width)
                    top = int(bbox["Top"] * height)
                    right = int((bbox["Left"] + bbox["Width"]) * width)
                    bottom = int((bbox["Top"] + bbox["Height"]) * height)

                    bounding_boxes.append(
                        OCRBoundingBox(
                            text=item["Text"],
                            coordinates=[
                                (left, top),
                                (right, top),
                                (right, bottom),
                                (left, bottom),
                            ],
                        )
                    )

        response = OCRResponse(
            all_text=extracted_text.strip(), bounding_boxes=bounding_boxes
        )
        return response

    def get_name(self) -> str:
        return "AWS Textract - OCR"

    @classmethod
    def init_from_env(
        cls, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None
    ) -> "AWSOCR":
        return cls(aws_access_key_id, aws_secret_access_key, aws_session_token)


# factory


def aws_provider_factory(task: Task, **config) -> ProviderHandler:
    aws_access_key_id = config.get("aws_access_key_id")
    aws_secret_access_key = config.get("aws_secret_access_key")
    aws_session_token = config.get("aws_session_token")

    if task not in AWS_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.SENTIMENT_ANALYSIS:
        return AWSSentimentAnalysis.init_from_env(
            aws_access_key_id, aws_secret_access_key, aws_session_token
        )
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return AWSTranslate.init_from_env(
            source_language=source_language,
            target_language=target_language,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
    elif task == Task.OCR:
        return AWSOCR.init_from_env(
            aws_access_key_id, aws_secret_access_key, aws_session_token
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

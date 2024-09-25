import boto3

from supercontrast.providers.provider import Provider
from supercontrast.tasks.ocr import OCRRequest, OCRResponse
from supercontrast.tasks.sentiment_analysis import SentimentAnalysisRequest, SentimentAnalysisResponse
from supercontrast.tasks.translation import TranslationRequest, TranslationResponse


def truncate_text(text: str, max_bytes: int = 5000):
    """Truncate text to a maximum of max_bytes bytes."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


class AWSSentimentAnalysis(Provider):
    def __init__(self):
        super().__init__()
        self.client = boto3.client("comprehend")
        self.THRESHOLD = 0

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        response = self.client.detect_sentiment(
            Text=truncate_text(request.text), 
            LanguageCode="en"
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


class AWSTranslate(Provider):
    def __init__(self, src_language: str, target_language: str):
        super().__init__()
        self.client = boto3.client("translate")
        self.src_language = src_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        response = self.client.translate_text(
            Text=truncate_text(request.text),
        )
        translated_text = response["TranslatedText"]
        
        result = TranslationResponse(
            text=translated_text,
        )

        return result

    def get_name(self) -> str:
        return "AWS Translate"

    @classmethod
    def init_from_env(cls, source_language: str, target_language: str) -> "AWSTranslate":
        return cls(source_language, target_language)


class AWSOCR(Provider):
    def __init__(self):
        super().__init__()
        self.client = boto3.client("textract")

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            with open(request.image, "rb") as image_file:
                image_data = image_file.read()
        else:
            image_data = request.image

        response = self.client.analyze_document(
            Document={"Bytes": image_data},
            FeatureTypes=["FORMS", "TABLES"]
        )
        
        extracted_text = ""
        for item in response.get('Blocks', []):
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + "\n"
        
        return OCRResponse(text=extracted_text.strip())

    def get_name(self) -> str:
        return "AWS Textract - OCR"

    @classmethod
    def init_from_env(cls) -> "AWSOCR":
        return cls()
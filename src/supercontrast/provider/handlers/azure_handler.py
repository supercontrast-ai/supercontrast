import azure.cognitiveservices.speech as speechsdk
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
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRBoundingBox, OCRRequest, OCRResponse
from supercontrast.task.types.sentiment_analysis_types import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from supercontrast.task.types.transcription_types import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from supercontrast.task.types.translation_types import (
    TranslationRequest,
    TranslationResponse,
)
from supercontrast.utils.audio import load_audio_file
from supercontrast.utils.image import get_image_size, load_image_data

# Constants

AZURE_SUPPORTED_TASKS = [
    Task.SENTIMENT_ANALYSIS,
    Task.TRANSLATION,
    Task.OCR,
    Task.TRANSCRIPTION,
]

# Task.SENTIMENT_ANALYSIS


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


# Task.TRANSLATION


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


# Task.OCR


class AzureOCR(ProviderHandler):
    def __init__(self, endpoint: str, key: str):
        super().__init__(provider=Provider.AZURE, task=Task.OCR)
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)
        read_response = self.client.read_in_stream(io.BytesIO(image_data), raw=True)

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
        bounding_boxes = []

        if read_result.status == OperationStatusCodes.succeeded:  # type: ignore
            for text_result in read_result.analyze_result.read_results:  # type: ignore
                for line in text_result.lines:
                    extracted_text += line.text + "\n"
                    bounding_boxes.append(
                        OCRBoundingBox(
                            text=line.text,
                            coordinates=[
                                (line.bounding_box[0], line.bounding_box[1]),
                                (line.bounding_box[2], line.bounding_box[3]),
                                (line.bounding_box[4], line.bounding_box[5]),
                                (line.bounding_box[6], line.bounding_box[7]),
                            ],
                        )
                    )

        response = OCRResponse(
            all_text=extracted_text.strip(), bounding_boxes=bounding_boxes
        )
        return response

    def get_name(self) -> str:
        return "Azure Computer Vision - OCR"

    @classmethod
    def init_from_env(cls, endpoint=None, key=None) -> "AzureOCR":
        endpoint = endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
        key = key or os.environ.get("AZURE_VISION_KEY")
        if not endpoint or not key:
            raise ValueError("AZURE_VISION_ENDPOINT and AZURE_VISION_KEY must be set")

        return cls(endpoint, key)


# Task.TRANSCRIPTION


class AzureTranscription(ProviderHandler):
    def __init__(self, speech_key: str, service_region: str):
        super().__init__(provider=Provider.AZURE, task=Task.TRANSCRIPTION)
        self.speech_key = speech_key
        self.service_region = service_region

    def request(self, request: TranscriptionRequest) -> TranscriptionResponse:
        audio_file_path = load_audio_file(request.audio_file)

        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.service_region
        )
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        complete_transcription = []
        done = False

        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                complete_transcription.append(evt.result.text)
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized")

        def stop_cb(evt):
            nonlocal done
            done = True

        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.session_started.connect(lambda evt: print("Session started"))

        speech_recognizer.start_continuous_recognition()

        while not done:
            time.sleep(0.5)

        speech_recognizer.stop_continuous_recognition()

        if audio_file_path != request.audio_file:
            os.unlink(audio_file_path)

        return TranscriptionResponse(text=" ".join(complete_transcription))

    def get_name(self) -> str:
        return "Azure Speech Services - Transcription"

    @classmethod
    def init_from_env(
        cls, speech_key=None, service_region=None
    ) -> "AzureTranscription":
        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        service_region = service_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_key or not service_region:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set")
        return cls(speech_key, service_region)


# factory


def azure_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in AZURE_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.SENTIMENT_ANALYSIS:
        endpoint = config.get("azure_text_analytics_endpoint")
        key = config.get("azure_text_analytics_key")
        return AzureSentimentAnalysis.init_from_env(endpoint=endpoint, key=key)
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        key = config.get("azure_text_analytics_key")
        region = config.get("azure_translator_region")
        return AzureTranslation.init_from_env(
            source_language=source_language,
            target_language=target_language,
            key=key,
            region=region,
        )
    elif task == Task.OCR:
        endpoint = config.get("azure_vision_endpoint")
        key = config.get("azure_vision_key")
        return AzureOCR.init_from_env(endpoint=endpoint, key=key)
    elif task == Task.TRANSCRIPTION:
        speech_key = config.get("azure_speech_key")
        service_region = config.get("azure_speech_region")
        return AzureTranscription.init_from_env(
            speech_key=speech_key, service_region=service_region
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

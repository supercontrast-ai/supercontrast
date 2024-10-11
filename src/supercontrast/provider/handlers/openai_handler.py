import os
import pydantic

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRRequest, OCRResponse
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
from supercontrast.utils.image import (
    get_image_size,
    load_image_data,
    process_image_for_llm,
)
from supercontrast.utils.text import truncate_text

# Constants

OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_SUPPORTED_TASKS = [
    # Task.OCR, # TODO: Add back in when request is fixed
    Task.SENTIMENT_ANALYSIS,
    Task.TRANSLATION,
    Task.TRANSCRIPTION,
]
# Task.SENTIMENT_ANALYSIS


class SentimentAnalysisOutput(pydantic.BaseModel):
    score: float = pydantic.Field(
        description="Sentiment score between -1 (very negative) and 1 (very positive)"
    )


SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    template="Analyze the sentiment of the following text. Respond with 1 for positive, 0 for neutral, and -1 for negative:\n\n{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=SentimentAnalysisOutput
        ).get_format_instructions()
    },
)


class OpenAISentimentAnalysis(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.OPENAI, task=Task.SENTIMENT_ANALYSIS)
        model = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": OPENAI_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)
        self.generator = SENTIMENT_ANALYSIS_PROMPT | model | parser

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        result: SentimentAnalysisOutput = self.generator.invoke(
            {"text": truncate_text(request.text)}
        )
        return SentimentAnalysisResponse(score=result.score)

    def get_name(self) -> str:
        return "OpenAI - Sentiment Analysis"

    @classmethod
    def init_from_env(cls) -> "OpenAISentimentAnalysis":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return cls()


# Task.TRANSLATION


class TranslationOutput(pydantic.BaseModel):
    translation: str = pydantic.Field(description="Translated text")


TRANSLATION_PROMPT = PromptTemplate(
    template="Translate the following text from {src_language} to {target_language}:\n\n{text}\n\n{format_instructions}",
    input_variables=["src_language", "target_language", "text"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=TranslationOutput
        ).get_format_instructions()
    },
)


class OpenAITranslate(ProviderHandler):
    def __init__(self, src_language: str, target_language: str):
        super().__init__(provider=Provider.OPENAI, task=Task.TRANSLATION)
        model = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": OPENAI_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=TranslationOutput)
        self.generator = TRANSLATION_PROMPT | model | parser
        self.src_language = src_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        result: TranslationOutput = self.generator.invoke(
            {
                "src_language": self.src_language,
                "target_language": self.target_language,
                "text": truncate_text(request.text),
            }
        )
        return TranslationResponse(text=result.translation)

    def get_name(self) -> str:
        return "OpenAI - Translate"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "OpenAITranslate":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return cls(source_language, target_language)


# Task.OCR


class OCROutput(pydantic.BaseModel):
    text: str = pydantic.Field(description="Extracted text from the image")


OCR_PROMPT = PromptTemplate(
    template="Return all text that is visible in the image:\n\n{image}\n\n{format_instructions}",
    input_variables=["image"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=OCROutput
        ).get_format_instructions()
    },
)


class OpenAIOCR(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.OPENAI, task=Task.OCR)
        model = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": OPENAI_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=OCROutput)
        self.generator = OCR_PROMPT | model | parser

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)
        processed_image = process_image_for_llm(image_data)
        if processed_image is None:
            raise ValueError("Failed to process the image")

        width, height = get_image_size(image_data)

        result: OCROutput = self.generator.invoke(
            {"image": f"data:image/jpeg;base64,{processed_image}"}
        )

        bounding_boxes = []

        return OCRResponse(all_text=result.text, bounding_boxes=bounding_boxes)

    def get_name(self) -> str:
        return "OpenAI - OCR"

    @classmethod
    def init_from_env(cls) -> "OpenAIOCR":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return cls()


# Task.TRANSCRIPTION


class OpenAITranscription(ProviderHandler):
    def __init__(self, api_key: str):
        super().__init__(provider=Provider.OPENAI, task=Task.TRANSCRIPTION)
        self.client = OpenAI(api_key=api_key)

    def request(self, request: TranscriptionRequest) -> TranscriptionResponse:
        audio_file_path = load_audio_file(request.audio_file)

        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="json"
            )

        if audio_file_path != request.audio_file:
            os.unlink(audio_file_path)  # Clean up the temporary file if it was created

        return TranscriptionResponse(text=transcript.text)

    def get_name(self) -> str:
        return "OpenAI Whisper - Transcription"

    @classmethod
    def init_from_env(cls, api_key=None) -> "OpenAITranscription":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        return cls(api_key)


# factory


def openai_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in OPENAI_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.SENTIMENT_ANALYSIS:
        return OpenAISentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return OpenAITranslate.init_from_env(source_language, target_language)
    elif task == Task.OCR:
        return OpenAIOCR.init_from_env()
    elif task == Task.TRANSCRIPTION:
        api_key = config.get("openai_api_key")
        return OpenAITranscription.init_from_env(api_key=api_key)
    else:
        raise ValueError(f"Unsupported task: {task}")

import os
import pydantic

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    Task,
    TranslationRequest,
    TranslationResponse,
)
from supercontrast.utils.text import truncate_text

# Constants

OPENAI_MODEL_NAME = "gpt-4o"

# Task.SENTIMENT_ANALYSIS


class SentimentAnalysisOutput(pydantic.BaseModel):
    score: float = pydantic.Field(
        description="Sentiment score between -1 (very negative) and 1 (very positive)"
    )


SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    template="Analyze the sentiment of the following text. Respond with a single number between -1 (very negative) and 1 (very positive):\n\n{text}\n\n{format_instructions}",
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


# factory


def openai_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.SENTIMENT_ANALYSIS:
        return OpenAISentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return OpenAITranslate.init_from_env(source_language, target_language)
    else:
        raise ValueError(f"Unsupported task: {task}")

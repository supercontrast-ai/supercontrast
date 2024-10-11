import os
import pydantic

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.sentiment_analysis_types import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from supercontrast.task.types.translation_types import (
    TranslationRequest,
    TranslationResponse,
)
from supercontrast.utils.text import truncate_text

# Constants

ANTHROPIC_MODEL_NAME = "claude-3-5-sonnet-20240620"
ANTHROPIC_SUPPORTED_TASKS = [Task.SENTIMENT_ANALYSIS, Task.TRANSLATION]

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


class AnthropicSentimentAnalysis(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.ANTHROPIC, task=Task.SENTIMENT_ANALYSIS)
        model = ChatAnthropic(
            **{
                "temperature": 0,
                "model": ANTHROPIC_MODEL_NAME,
            }
        )
        pydantic_parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)
        self.parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=model)
        self.generator = SENTIMENT_ANALYSIS_PROMPT | model | self.parser

    def request(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        result: SentimentAnalysisOutput = self.generator.invoke(
            {"text": truncate_text(request.text)}
        )
        return SentimentAnalysisResponse(score=result.score)

    def get_name(self) -> str:
        return "Anthropic - Sentiment Analysis"

    @classmethod
    def init_from_env(cls) -> "AnthropicSentimentAnalysis":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
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


class AnthropicTranslate(ProviderHandler):
    def __init__(self, src_language: str, target_language: str):
        super().__init__(provider=Provider.ANTHROPIC, task=Task.TRANSLATION)
        model = ChatAnthropic(
            **{
                "temperature": 0,
                "model": ANTHROPIC_MODEL_NAME,
            }
        )
        pydantic_parser = PydanticOutputParser(pydantic_object=TranslationOutput)
        self.parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=model)
        self.generator = TRANSLATION_PROMPT | model | self.parser
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
        return "Anthropic - Translate"

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "AnthropicTranslate":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        return cls(source_language, target_language)


# factory


def anthropic_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in ANTHROPIC_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.SENTIMENT_ANALYSIS:
        return AnthropicSentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return AnthropicTranslate.init_from_env(source_language, target_language)
    else:
        raise ValueError(f"Unsupported task: {task}")

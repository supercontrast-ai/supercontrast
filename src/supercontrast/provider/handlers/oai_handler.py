import os
import requests
import tempfile

from openai import OpenAI

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.transcription_types import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from supercontrast.utils.audio import load_audio_file


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


def openai_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.TRANSCRIPTION:
        api_key = config.get("openai_api_key")
        return OpenAITranscription.init_from_env(api_key=api_key)
    else:
        raise ValueError(f"Unsupported task: {task}")

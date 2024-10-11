import os
import requests

from typing import Any, Dict, List, Optional

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.translation_types import (
    TranslationRequest,
    TranslationResponse,
)

# Constants

MODERNMT_SUPPORTED_TASKS = [Task.TRANSLATION]

# Task.TRANSLATION


class ModernMTTranslation(ProviderHandler):
    def __init__(self, api_key: str, source_language: str, target_language: str):
        super().__init__(provider=Provider.MODERNMT, task=Task.TRANSLATION)
        self.__base_url = "https://api.modernmt.com"
        self.__headers = {
            "MMT-ApiKey": api_key,
            "MMT-Platform": "modernmt-python",
            "MMT-PlatformVersion": "1.5.2",
        }
        self.source_language = source_language
        self.target_language = target_language

    def request(self, request: TranslationRequest) -> TranslationResponse:
        data = {
            "source": self.source_language,
            "target": self.target_language,
            "q": request.text,
        }

        response = self.__send("get", "/translate", data=data)

        if isinstance(response, list):
            translated_text = response[0]["translation"]
        else:
            translated_text = response["translation"]

        return TranslationResponse(text=translated_text)

    def get_name(self) -> str:
        return "ModernMT Translation"

    def __send(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Any:
        url = self.__base_url + endpoint

        headers = self.__headers.copy()
        headers["X-HTTP-Method-Override"] = method

        r = requests.post(url, headers=headers, json=data)

        if r.status_code != requests.codes.ok:
            raise ModernMTException(r.status_code, "TranslationError", r.text)

        return r.json()["data"]

    @classmethod
    def init_from_env(
        cls, source_language: str, target_language: str
    ) -> "ModernMTTranslation":
        api_key = os.environ.get("MODERN_MT_API_KEY")
        if not api_key:
            raise EnvironmentError("MODERN_MT_API_KEY environment variable is not set")
        return cls(api_key, source_language, target_language)


class ModernMTException(Exception):
    def __init__(self, status: int, type: str, message: str) -> None:
        super().__init__(f"({type}) {message}")
        self.status = status
        self.type = type
        self.message = message


def modernmt_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in MODERNMT_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        return ModernMTTranslation.init_from_env(
            source_language=source_language,
            target_language=target_language,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

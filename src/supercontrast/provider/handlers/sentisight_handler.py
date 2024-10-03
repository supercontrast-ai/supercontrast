import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import requests

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task import (
    OCRRequest,
    OCRResponse,
    Task,
)

class Point(BaseModel):
    x: int
    y: int

class TextSegment(BaseModel):
    label: str
    score: float
    points: List[Point]

class OCRResult(BaseModel):
    segments: List[TextSegment]

class SentisightOCR(ProviderHandler):
    def __init__(self, api_key: str, language: str):
        super().__init__(provider=Provider.SENTISIGHT, task=Task.OCR)
        self.key = api_key
        self.language = language
        self.base_url = "https://platform.sentisight.ai/api/pm-predict/"

    def request(self, request: OCRRequest) -> OCRResponse:
        if isinstance(request.image, str):
            if request.image.startswith("http") or request.image.startswith("https"):
                image_data = requests.get(request.image).content
            else:
                with open(request.image, "rb") as file_:
                    image_data = file_.read()
        elif isinstance(request.image, bytes):
            image_data = request.image
        else:
            raise ValueError("Invalid image type")


        url = f"{self.base_url}Text-recognition"

        response = requests.post(
            url=f"{url}?lang={self.language}",
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
                "Content-Type": "application/octet-stream",
            },
            data=image_data,
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        response_data: OCRResult = OCRResult(segments=[TextSegment(**item) for item in response.json()])

        text = ""
        for segment in response_data.segments:
            text += segment.label + "\n"

        return OCRResponse(text=text)

    def get_name(self) -> str:
        return "Sentisight OCR"

    @classmethod
    def init_from_env(cls, language: str) -> "SentisightOCR":
        api_key = os.environ.get("SENTISIGHT_API_TOKEN")
        if not api_key:
            raise EnvironmentError("SENTISIGHT_API_TOKEN environment variable is not set")
        return cls(api_key, language)

def sentisight_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.OCR:
        language = config.get("language", "en")
        return SentisightOCR.init_from_env(language=language)
    else:
        raise ValueError(f"Unsupported task: {task}")

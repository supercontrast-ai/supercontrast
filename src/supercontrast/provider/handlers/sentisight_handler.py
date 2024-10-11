import os
import requests

from pydantic import BaseModel
from typing import List

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRBoundingBox, OCRRequest, OCRResponse
from supercontrast.utils.image import load_image_data

# Constants

SENTISIGHT_SUPPORTED_TASKS = [Task.OCR]

# Task.OCR


class Point(BaseModel):
    x: int
    y: int


class TextSegment(BaseModel):
    label: str
    score: float
    points: List[Point]


class SentisightOCRResult(BaseModel):
    segments: List[TextSegment]


class SentisightOCR(ProviderHandler):
    def __init__(self, api_key: str, language: str):
        super().__init__(provider=Provider.SENTISIGHT, task=Task.OCR)
        self.key = api_key
        self.language = language
        self.base_url = "https://platform.sentisight.ai/api/pm-predict/"

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)

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

        response_data: SentisightOCRResult = SentisightOCRResult(
            segments=[TextSegment(**item) for item in response.json()]
        )

        text = ""
        for segment in response_data.segments:
            text += segment.label + "\n"

        response = OCRResponse(
            all_text=text,
            bounding_boxes=[
                OCRBoundingBox(
                    text=segment.label,
                    coordinates=[(point.x, point.y) for point in segment.points],
                )
                for segment in response_data.segments
            ],
        )
        return response

    def get_name(self) -> str:
        return "Sentisight OCR"

    @classmethod
    def init_from_env(cls, language: str) -> "SentisightOCR":
        api_key = os.environ.get("SENTISIGHT_API_TOKEN")
        if not api_key:
            raise EnvironmentError(
                "SENTISIGHT_API_TOKEN environment variable is not set"
            )
        return cls(api_key, language)


def sentisight_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in SENTISIGHT_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.OCR:
        language = config.get("language", "en")
        return SentisightOCR.init_from_env(language=language)
    else:
        raise ValueError(f"Unsupported task: {task}")

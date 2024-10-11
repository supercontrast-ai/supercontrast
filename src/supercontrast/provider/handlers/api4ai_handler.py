import os
import requests

from pydantic import BaseModel
from typing import Dict, List

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRBoundingBox, OCRRequest, OCRResponse
from supercontrast.utils.image import load_image_data

# Constants

API4AI_SUPPORTED_TASKS = [Task.OCR]

# Task.OCR


class Point(BaseModel):
    x: int
    y: int


class TextSegment(BaseModel):
    text: str
    score: float
    points: List[Point]


class Entity(BaseModel):
    kind: str
    name: str
    text: str


class Object(BaseModel):
    box: List[float]
    entities: List[Entity]


class EntityGroup(BaseModel):
    kind: str
    name: str
    objects: List[Object]


class Result(BaseModel):
    status: Dict[str, str]
    name: str
    md5: str
    width: int
    height: int
    entities: List[EntityGroup]


class OCRResult(BaseModel):
    results: List[Result]


class API4AIOCR(ProviderHandler):
    def __init__(self, api_key: str):
        super().__init__(provider=Provider.API4AI, task=Task.OCR)
        self.key = api_key
        self.url = "https://ocr43.p.rapidapi.com/v1/results"
        self.headers = {"X-RapidAPI-Key": self.key}

    def request(self, request: OCRRequest) -> OCRResponse:
        image_data = load_image_data(request.image)
        files = {"image": image_data}

        response = requests.post(url=self.url, files=files, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        response_json = response.json()
        response_data = OCRResult(**response_json)

        all_text = ""
        bounding_boxes = []
        for result in response_data.results:
            for entity in result.entities:
                for object in entity.objects:
                    for entity in object.entities:
                        all_text += entity.text + " "
                        coordinates = [
                            (
                                int(object.box[0] * result.width),
                                int(object.box[1] * result.height),
                            ),
                            (
                                int(object.box[2] * result.width),
                                int(object.box[1] * result.height),
                            ),
                            (
                                int(object.box[2] * result.width),
                                int(object.box[3] * result.height),
                            ),
                            (
                                int(object.box[0] * result.width),
                                int(object.box[3] * result.height),
                            ),
                        ]
                        bounding_boxes.append(
                            OCRBoundingBox(text=entity.text, coordinates=coordinates)
                        )

        return OCRResponse(all_text=all_text.strip(), bounding_boxes=bounding_boxes)

    def get_name(self) -> str:
        return "API4AI OCR"

    @classmethod
    def init_from_env(cls) -> "API4AIOCR":
        api_key = os.environ.get("API4AI_API_KEY", "")
        if not api_key:
            raise ValueError("API4AI_API_KEY is not set")
        return cls(api_key)


# factory


def api4ai_provider_factory(task: Task, **config) -> ProviderHandler:
    if task not in API4AI_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.OCR:
        return API4AIOCR.init_from_env()
    else:
        raise ValueError(f"Unsupported task: {task}")

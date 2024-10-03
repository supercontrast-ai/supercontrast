import os
import requests

from pydantic import BaseModel, Field
from typing import Dict, List

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task import OCRRequest, OCRResponse, Task


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

        files = {"image": image_data}

        response = requests.post(url=self.url, files=files, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        response_json = response.json()
        response_data = OCRResult(**response_json)

        text = ""
        for result in response_data.results:
            for entity in result.entities:
                for object in entity.objects:
                    for entity in object.entities:
                        text += entity.text

        return OCRResponse(text=text)

    def get_name(self) -> str:
        return "API4AI OCR"

    @classmethod
    def init_from_env(cls) -> "API4AIOCR":
        api_key = os.environ.get("API4AI_API_KEY", "")
        if not api_key:
            raise ValueError("API4AI_API_KEY is not set")
        return cls(api_key)


def api4ai_provider_factory(task: Task, **config) -> ProviderHandler:
    if task == Task.OCR:
        return API4AIOCR.init_from_env()
    else:
        raise ValueError(f"Unsupported task: {task}")

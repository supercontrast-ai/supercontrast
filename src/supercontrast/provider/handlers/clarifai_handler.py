import os
import requests

from pydantic import BaseModel, Field
from typing import List, Optional

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task import OCRRequest, OCRResponse, Task


class TextInfo(BaseModel):
    encoding: str


class Text(BaseModel):
    raw: str
    text_info: TextInfo


class BoundingBox(BaseModel):
    top_row: float
    left_col: float
    bottom_row: float
    right_col: float


class RegionInfo(BaseModel):
    bounding_box: BoundingBox


class Region(BaseModel):
    id: str
    region_info: RegionInfo
    data: dict
    value: float


class Output(BaseModel):
    id: str
    status: dict
    created_at: str
    model: dict
    input: dict
    data: dict = Field(..., alias="data")

    @property
    def regions(self) -> List[Region]:
        return [Region(**region) for region in self.data.get("regions", [])]


class ClarifaiResponse(BaseModel):
    status: dict
    outputs: List[Output]


class ClarifaiOCR(ProviderHandler):
    model_name = "ocr-scene-english-paddleocr"
    model_version = "46e99516c2d94f58baf2bcaf5a6a53a9"

    def __init__(self, api_key):
        super().__init__(provider=Provider.CLARIFAI, task=Task.OCR)
        self.api_key = api_key
        self.base_url = f"https://api.clarifai.com/v2/users/clarifai/apps/main/models/{self.model_name}/versions/{self.model_version}/outputs"

    def request(self, request: OCRRequest) -> OCRResponse:
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": [
                {
                    "data": {
                        "image": {
                            "url": request.image
                            if isinstance(request.image, str)
                            else None
                        }
                    }
                }
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        parsed_response = ClarifaiResponse(**result)
        text = self.extract_text(parsed_response)
        return OCRResponse(text=text)

    def extract_text(self, response: ClarifaiResponse) -> str:
        text_parts = []
        for output in response.outputs:
            for region in output.regions:
                if "text" in region.data and "raw" in region.data["text"]:
                    text_parts.append(region.data["text"]["raw"])
        return " ".join(text_parts)

    def get_name(self) -> str:
        return "Clarifai - OCR"

    @classmethod
    def init_from_env(cls, api_key) -> "ClarifaiOCR":
        return cls(api_key)


def clarifai_provider_factory(task: Task, **config) -> ProviderHandler:
    api_key = os.environ.get("CLARIFAI_API_KEY")

    if api_key is None:
        raise ValueError("CLARIFAI_API_KEY is not set")

    if task == Task.OCR:
        return ClarifaiOCR.init_from_env(api_key)
    else:
        raise ValueError(f"Unsupported task: {task}")

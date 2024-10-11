import os
import requests

from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from typing import List

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRBoundingBox, OCRRequest, OCRResponse
from supercontrast.utils.image import get_image_size, load_image_data

# Constants

CLARIFAI_SUPPORTED_TASKS = [
    # Task.OCR, # TODO: Add back in when request is fixed
]

# Task.OCR


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

    def get_image_size(self, image):
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(BytesIO(image))
        else:
            raise ValueError("Unsupported image type")

        return img.size

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
                            "url": (
                                request.image
                                if isinstance(request.image, str)
                                else None
                            )
                        }
                    }
                }
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        parsed_response = ClarifaiResponse(**result)

        # Get image size
        image_size = get_image_size(request.image)

        text_parts = []
        bounding_boxes = []
        for output in parsed_response.outputs:
            for region in output.regions:
                if "text" in region.data and "raw" in region.data["text"]:
                    text = region.data["text"]["raw"]
                    text_parts.append(text)
                    bbox = region.region_info.bounding_box
                    coordinates = [
                        (
                            int(bbox.left_col * image_size[0]),
                            int(bbox.top_row * image_size[1]),
                        ),
                        (
                            int(bbox.right_col * image_size[0]),
                            int(bbox.top_row * image_size[1]),
                        ),
                        (
                            int(bbox.right_col * image_size[0]),
                            int(bbox.bottom_row * image_size[1]),
                        ),
                        (
                            int(bbox.left_col * image_size[0]),
                            int(bbox.bottom_row * image_size[1]),
                        ),
                    ]
                    bounding_boxes.append(
                        OCRBoundingBox(text=text, coordinates=coordinates)
                    )

        all_text = " ".join(text_parts)
        response = OCRResponse(all_text=all_text, bounding_boxes=bounding_boxes)
        return response

    def get_name(self) -> str:
        return "Clarifai - OCR"

    @classmethod
    def init_from_env(cls, api_key) -> "ClarifaiOCR":
        return cls(api_key)


# factory


def clarifai_provider_factory(task: Task, **config) -> ProviderHandler:
    api_key = os.environ.get("CLARIFAI_API_KEY")

    if api_key is None:
        raise ValueError("CLARIFAI_API_KEY is not set")

    if task not in CLARIFAI_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.OCR:
        return ClarifaiOCR.init_from_env(api_key)
    else:
        raise ValueError(f"Unsupported task: {task}")

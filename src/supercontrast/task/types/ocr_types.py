from pydantic import BaseModel
from typing import List, Optional, Tuple, Union

# Request


class OCRRequest(BaseModel):
    image: Union[str, bytes]


# Response


class OCRBoundingBox(BaseModel):
    text: str
    coordinates: List[Tuple[int, int]]


class OCRResponse(BaseModel):
    all_text: str
    bounding_boxes: List[OCRBoundingBox]

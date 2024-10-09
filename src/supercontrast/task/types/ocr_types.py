from pydantic import BaseModel
from typing import List, Tuple, Union

# Request


class OCRRequest(BaseModel):
    image: Union[str, bytes]


# Response


class OCRBoundingBox(BaseModel):
    text: str
    coordinates: List[Tuple[int, int]]

    def __str__(self):
        return f"OCRBoundingBox(text: '{self.text}', coordinates: {self.coordinates})"


class OCRResponse(BaseModel):
    all_text: str
    bounding_boxes: List[OCRBoundingBox]

    def __str__(self):
        bounding_boxes_str = "\n    ".join(str(box) for box in self.bounding_boxes)
        return f"OCRResponse(\n  all_text: '{self.all_text}',\n  bounding_boxes: [\n    {bounding_boxes_str}\n  ]\n)"

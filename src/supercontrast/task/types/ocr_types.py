from pydantic import BaseModel, field_validator
from typing import List, Optional, Tuple, Union

# Request


class OCRRequest(BaseModel):
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    pdf_path: Optional[str] = None
    pdf_data: Optional[bytes] = None

    @field_validator('image_path', 'image_data', 'pdf_path', 'pdf_data', always=True)
    def check_image_or_pdf(cls, v, values):
        if not any([values.get('image_path'), values.get('image_data'),
                    values.get('pdf_path'), values.get('pdf_data')]):
            raise ValueError('Either image or pdf must be provided')
        return v


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

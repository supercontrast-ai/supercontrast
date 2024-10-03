from pydantic import BaseModel
from typing import Union

# Request


class OCRRequest(BaseModel):
    image: Union[str, bytes]


# Response


class OCRResponse(BaseModel):
    text: str

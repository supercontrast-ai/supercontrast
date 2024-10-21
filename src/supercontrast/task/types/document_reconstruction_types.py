from pydantic import BaseModel
from typing import Any


class DocumentReconstructionRequest(BaseModel):
    input_file: str


class DocumentReconstructionResponse(BaseModel):
    output: Any

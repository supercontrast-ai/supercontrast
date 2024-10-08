from pydantic import BaseModel

# Request


class TranscriptionRequest(BaseModel):
    audio_file: str


# Response


class TranscriptionResponse(BaseModel):
    text: str

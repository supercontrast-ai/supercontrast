from pydantic import BaseModel

# Request


class TranslationRequest(BaseModel):
    text: str


# Response


class TranslationResponse(BaseModel):
    text: str

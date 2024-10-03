from pydantic import BaseModel

# Request


class SentimentAnalysisRequest(BaseModel):
    text: str


# Response


class SentimentAnalysisResponse(BaseModel):
    score: float

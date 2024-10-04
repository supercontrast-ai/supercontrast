from enum import Enum


class Provider(Enum):
    # major providers
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    SENTISIGHT = "sentisight"
    API4AI = "api4ai"
    CLARIFAI = "clarifai"
    MODERNMT = "modernmt"
    OPENAI = "openai"

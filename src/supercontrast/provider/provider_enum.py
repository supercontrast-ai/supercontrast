from enum import Enum


class Provider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    SENTISIGHT = "sentisight"
    API4AI = "api4ai"
    CLARIFAI = "clarifai"
    MODERNMT = "modernmt"
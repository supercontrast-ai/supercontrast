from enum import Enum


class Task(Enum):
    DOCUMENT_RECONSTRUCTION = "document_reconstruction"
    OCR = "ocr"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"

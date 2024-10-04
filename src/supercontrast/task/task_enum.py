from enum import Enum


class Task(Enum):
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TRANSLATION = "translation"
    OCR = "ocr"
    TRANSCRIPTION = "transcription"

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import (
    OCRRequest,
    SentimentAnalysisRequest,
    Task,
    TranslationRequest,
)

azure_translation_client = supercontrast_client(
    task=Task.TRANSLATION,
    providers=[Provider.AZURE],
    source_language="en",
    target_language="es",
)

request = TranslationRequest(text="Hello, world!")
response = azure_translation_client.request(request)

print("-" * 80)
print("Translation Request:")
print(request)
print("Translation Response from Azure:")
print(response)
print("-" * 80)

aws_sentiment_analysis_client = supercontrast_client(
    task=Task.SENTIMENT_ANALYSIS,
    providers=[Provider.AWS],
)

request = SentimentAnalysisRequest(text="I love programming in Python!")
response = aws_sentiment_analysis_client.request(request)

print("-" * 80)
print("Sentiment Analysis Request:")
print(request)
print("Sentiment Analysis Response from AWS:")
print(response)
print("-" * 80)

gcp_ocr_client = supercontrast_client(
    task=Task.OCR,
    providers=[Provider.GCP],
)

request = OCRRequest(image="./example_data/ocr_sample.png")
response = gcp_ocr_client.request(request)
print("-" * 80)
print("OCR Request (includes an image with the text 'I'm sample text')")
print("OCR Response from GCP:")
print(response)
print("-" * 80)

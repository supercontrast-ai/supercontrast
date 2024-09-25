from supercontrast.client import supercontrast_client
from supercontrast.providers.provider import ProviderType
from supercontrast.tasks.ocr import OCRRequest
from supercontrast.tasks.sentiment_analysis import SentimentAnalysisRequest
from supercontrast.tasks.task_types import Task
from supercontrast.tasks.translation import TranslationRequest

azure_translation_client = supercontrast_client(
    task=Task.TRANSLATION,
    providers=[ProviderType.AZURE],
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
    providers=[ProviderType.AWS],
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
    providers=[ProviderType.GCP],
)

request = OCRRequest(image="./example_data/ocr_sample.png")
response = gcp_ocr_client.request(request)
print("-" * 80)
print("OCR Request (includes an image with the text 'I'm sample text')")
print("OCR Response from GCP:")
print(response)
print("-" * 80)

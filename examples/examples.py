from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRRequest
from supercontrast.task.types.sentiment_analysis_types import SentimentAnalysisRequest
from supercontrast.task.types.translation_types import TranslationRequest


def print_response(provider, task, input_data, response):
    print("-" * 100)
    print(f"{provider} {task} Response:")
    print(f"Input {'Image' if task == Task.OCR else 'Text'}: {input_data}")
    print(f"Response: \n\t{response}")
    print("-" * 100)


# Sending a OCR Request to GCP
client = supercontrast_client(task=Task.OCR, providers=[Provider.GCP], optimizer=None)
input_image = "https://jeroen.github.io/images/testocr.png"
response = client.request(OCRRequest(image=input_image))
print_response("GCP", Task.OCR, input_image, response)

# Sending a Sentiment Analysis Request to AWS
client = supercontrast_client(
    task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS], optimizer=None
)
input_text = "I love this product!"
response = client.request(SentimentAnalysisRequest(text=input_text))
print_response("AWS", Task.SENTIMENT_ANALYSIS, input_text, response)

# Sending a Translation Request to Azure
client = supercontrast_client(
    task=Task.TRANSLATION,
    providers=[Provider.AZURE],
    optimizer=None,
    source_language="en",
    target_language="fr",
)
input_text = "I love this product!"
response = client.request(TranslationRequest(text=input_text))
print_response("Azure", Task.TRANSLATION, input_text, response)

# Sending a OCR Request to Sentisight
client = supercontrast_client(
    task=Task.OCR,
    providers=[Provider.SENTISIGHT],
    optimizer=None,
    language="en",
)
input_image = "https://jeroen.github.io/images/testocr.png"
response = client.request(OCRRequest(image=input_image))
print_response("Sentisight", Task.OCR, input_image, response)

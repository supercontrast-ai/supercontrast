from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRRequest
from supercontrast.task.types.sentiment_analysis_types import SentimentAnalysisRequest
from supercontrast.task.types.transcription_types import TranscriptionRequest
from supercontrast.task.types.translation_types import TranslationRequest


def print_response(provider, task, input_data, response):
    print("-" * 100)
    print(f"{provider} {task} Response:")
    print(f"Input {'Image' if task == Task.OCR else 'Text'}: {input_data}")
    print(f"Response: \n\t{response}")
    print("-" * 100)


# # Sending a OCR Request to GCP
# client = supercontrast_client(task=Task.OCR, providers=[Provider.GCP], optimizer=None)
# input_image = "https://jeroen.github.io/images/testocr.png"
# response = client.request(OCRRequest(image=input_image))
# print_response("GCP", Task.OCR, input_image, response)

# # Sending a Sentiment Analysis Request to AWS
# client = supercontrast_client(
#     task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS], optimizer=None
# )
# input_text = "I love this product!"
# response = client.request(SentimentAnalysisRequest(text=input_text))
# print_response("AWS", Task.SENTIMENT_ANALYSIS, input_text, response)

# # Sending a Translation Request to Azure
# client = supercontrast_client(
#     task=Task.TRANSLATION,
#     providers=[Provider.AZURE],
#     optimizer=None,
#     source_language="en",
#     target_language="fr",
# )
# input_text = "I love this product!"
# response = client.request(TranslationRequest(text=input_text))
# print_response("Azure", Task.TRANSLATION, input_text, response)

# # Sending a OCR Request to Sentisight
# client = supercontrast_client(
#     task=Task.OCR,
#     providers=[Provider.SENTISIGHT],
#     optimizer=None,
#     language="en",
# )
# input_image = "https://jeroen.github.io/images/testocr.png"
# response = client.request(OCRRequest(image=input_image))
# print_response("Sentisight", Task.OCR, input_image, response)

# # Sending a OCR Request to API4AI
# client = supercontrast_client(
#     task=Task.OCR,
#     providers=[Provider.API4AI],
#     optimizer=None,
#     language="en",
# )
# input_image = "https://jeroen.github.io/images/testocr.png"
# response = client.request(OCRRequest(image=input_image))
# print_response("API4AI", Task.OCR, input_image, response)

# client = supercontrast_client(
#     task=Task.OCR,
#     providers=[Provider.CLARIFAI],
#     optimizer=None,
#     language="en",
# )
# input_image = "https://jeroen.github.io/images/testocr.png"
# request = OCRRequest(image=input_image)
# response = client.request(request)
# print_response("Clarifai", Task.OCR, input_image, response)


# client = supercontrast_client(
#     task=Task.TRANSLATION,
#     providers=[Provider.MODERNMT],
#     optimizer=None,
#     source_language="en",
#     target_language="fr",
# )
# input_text = "I love this product!"
# response = client.request(TranslationRequest(text=input_text))
# print_response("ModernMT", Task.TRANSLATION, input_text, response)

input_audio = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"

# client = supercontrast_client(
#     task=Task.TRANSCRIPTION,
#     providers=[Provider.OPENAI],
#     optimizer=None,
# )

# response = client.request(TranscriptionRequest(audio_file=input_audio))
# print_response("OpenAI", Task.TRANSCRIPTION, input_audio, response)

client = supercontrast_client(
    task=Task.TRANSCRIPTION,
    providers=[Provider.AZURE],
    optimizer=None,
)
response = client.request(TranscriptionRequest(audio_file=input_audio))
print_response("Azure", Task.TRANSCRIPTION, input_audio, response)

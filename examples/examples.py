import logging
import asyncio
from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRRequest
from supercontrast.task.types.sentiment_analysis_types import SentimentAnalysisRequest
from supercontrast.task.types.transcription_types import TranscriptionRequest
from supercontrast.task.types.translation_types import TranslationRequest

# Setup logging
logging.basicConfig(level=logging.INFO)

def create_client(task, provider, **kwargs):
    """Create a supercontrast client for the given task and provider."""
    return supercontrast_client(task=task, providers=[provider], **kwargs)

async def process_request(client, request, provider, task, input_data):
    """Process a request and log the response."""
    try:
        response = await asyncio.to_thread(client.request, request)
        logging.info(f"{provider} {task} Response:\n\tInput: {input_data}\n\tResponse: {response}")
    except Exception as e:
        logging.error(f"Error processing {task} for provider {provider}: {e}")

async def main():
    # OCR Request Examples
    input_image = "https://jeroen.github.io/images/testocr.png"
    ocr_providers = [Provider.GCP, Provider.SENTISIGHT, Provider.API4AI, Provider.CLARIFAI]
    ocr_tasks = [
        process_request(create_client(Task.OCR, provider), OCRRequest(image=input_image), provider.name, Task.OCR, input_image)
        for provider in ocr_providers
    ]

    # Sentiment Analysis Request Example
    input_text = "I love this product!"
    sentiment_client = create_client(Task.SENTIMENT_ANALYSIS, Provider.AWS)
    sentiment_task = process_request(sentiment_client, SentimentAnalysisRequest(text=input_text), "AWS", Task.SENTIMENT_ANALYSIS, input_text)

    # Translation Request Example
    translation_client = create_client(
        Task.TRANSLATION,
        Provider.AZURE,
        source_language="en",
        target_language="fr"
    )
    translation_task = process_request(translation_client, TranslationRequest(text=input_text), "Azure", Task.TRANSLATION, input_text)

    # Transcription Request Example
    input_audio = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"
    transcription_providers = [Provider.OPENAI, Provider.AZURE]
    transcription_tasks = [
        process_request(create_client(Task.TRANSCRIPTION, provider), TranscriptionRequest(audio_file=input_audio), provider.name, Task.TRANSCRIPTION, input_audio)
        for provider in transcription_providers
    ]

    # Run all tasks concurrently
    await asyncio.gather(*ocr_tasks, sentiment_task, translation_task, *transcription_tasks)

if __name__ == "__main__":
    asyncio.run(main())

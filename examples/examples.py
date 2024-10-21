from supercontrast import (
    DocumentReconstructionRequest,
    OCRRequest,
    Provider,
    SentimentAnalysisRequest,
    SuperContrastClient,
    Task,
    TranscriptionRequest,
    TranslationRequest,
)

# Constants
SAMPLE_AUDIO_URL = "https://github.com/supercontrast-ai/supercontrast/raw/main/tests/audio/test_transcription.wav"
SAMPLE_IMAGE_URL = "https://github.com/supercontrast-ai/supercontrast/raw/main/tests/image/test_ocr.png"
SAMPLE_PDF_URL = "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf"
SAMPLE_TEXT = "I love programming in Python!"


# Print response
def print_response(input_data, response, metadata):
    """Helper function to print responses in a consistent format."""
    print("-" * 100)
    print(f"Input: {input_data}")
    print(f"{metadata}")
    print(f"Response: \n{response}\n")
    print("-" * 100)


# Document Reconstruction Example
def run_document_reconstruction_example():
    """
    Demonstrates Document Reconstruction using the OmniAI provider.
    """
    print("\nRunning Document Reconstruction Example with OmniAI")
    client = SuperContrastClient(
        task=Task.DOCUMENT_RECONSTRUCTION, providers=[Provider.OMNIAI]
    )
    request = DocumentReconstructionRequest(input_file=SAMPLE_PDF_URL)
    response, metadata = client.request(request)
    print_response(SAMPLE_PDF_URL, response, metadata)


# OCR Example
def run_ocr_example():
    """
    Demonstrates Optical Character Recognition (OCR) using the GCP provider.
    This function sends an image URL to the GCP OCR service and retrieves the extracted text.
    It showcases how to initialize the SuperContrastClient for OCR tasks and handle the response.
    """
    print("\nRunning OCR Example with GCP")
    client = SuperContrastClient(task=Task.OCR, providers=[Provider.GCP])
    request = OCRRequest(image=SAMPLE_IMAGE_URL)
    response, metadata = client.request(request)
    print_response(SAMPLE_IMAGE_URL, response, metadata)


# Sentiment Analysis Example
def run_sentiment_analysis_example():
    """
    Demonstrates Sentiment Analysis using the AWS provider.
    This function sends a sample text to the AWS sentiment analysis service
    and retrieves the sentiment score. It shows how to set up the SuperContrastClient
    for sentiment analysis tasks and interpret the results.
    """
    print("\nRunning Sentiment Analysis Example with AWS")
    client = SuperContrastClient(task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS])
    request = SentimentAnalysisRequest(text=SAMPLE_TEXT)
    response, metadata = client.request(request)
    print_response(SAMPLE_TEXT, response, metadata)


# Transcription Example
def run_transcription_example():
    """
    Demonstrates Audio Transcription using the OpenAI provider.
    This function sends an audio file URL to OpenAI's transcription service
    and retrieves the transcribed text. It demonstrates how to set up the
    SuperContrastClient for audio transcription tasks and process the results.
    """
    print("\nRunning Transcription Example with OpenAI")
    client = SuperContrastClient(task=Task.TRANSCRIPTION, providers=[Provider.OPENAI])
    request = TranscriptionRequest(audio_file=SAMPLE_AUDIO_URL)
    response, metadata = client.request(request)
    print_response(SAMPLE_AUDIO_URL, response, metadata)


# Translation Example
def run_translation_example():
    """
    Demonstrates Text Translation using the Azure provider.
    This function translates a sample English text to French using Azure's translation service.
    It illustrates how to configure the SuperContrastClient for translation tasks,
    including setting source and target languages.
    """
    print("\nRunning Translation Example with Azure")
    client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.AZURE],
        source_language="en",
        target_language="fr",
    )
    request = TranslationRequest(text=SAMPLE_TEXT)
    response, metadata = client.request(request)
    print_response(SAMPLE_TEXT, response, metadata)


# Run all examples
if __name__ == "__main__":
    run_document_reconstruction_example()
    # run_ocr_example()
    # run_sentiment_analysis_example()
    # run_transcription_example()
    # run_translation_example()

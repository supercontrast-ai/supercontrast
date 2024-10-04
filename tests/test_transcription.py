from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, TranscriptionRequest, TranscriptionResponse

# constants

TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"

# helper functions


def print_request_and_response(
    request: TranscriptionRequest, response: TranscriptionResponse, provider: Provider
):
    print("\n", "-" * 80, "\n")
    print("Transcription Request:")
    print(request, "\n")
    print(f"Transcription Response from {provider}:")
    print(response, "\n")
    print("-" * 80, "\n")


# tests


def test_azure_transcription():
    transcription_azure_client = supercontrast_client(
        task=Task.TRANSCRIPTION, providers=[Provider.AZURE]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    response = transcription_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    print_request_and_response(request, response, provider=Provider.AZURE)


def test_openai_transcription():
    transcription_openai_client = supercontrast_client(
        task=Task.TRANSCRIPTION, providers=[Provider.OPENAI]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    response = transcription_openai_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    print_request_and_response(request, response, provider=Provider.OPENAI)

from supercontrast import (
    Client,
    Provider,
    Task,
    TaskMetadata,
    TranscriptionRequest,
    TranscriptionResponse,
)

# constants

TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"

# helper functions


def print_request_response_and_metadata(
    request: TranscriptionRequest,
    response: TranscriptionResponse,
    metadata: TaskMetadata,
):
    print("\n", "-" * 80, "\n")
    print("Transcription Request:")
    print(request, "\n")
    print(f"Transcription Response from {metadata.provider}:")
    print(response, "\n")
    print("Metadata:")
    print(metadata, "\n")
    print("-" * 80, "\n")


# tests


def test_azure_transcription():
    transcription_azure_client = Client(
        task=Task.TRANSCRIPTION, providers=[Provider.AZURE]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    response, metadata = transcription_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSCRIPTION
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_openai_transcription():
    transcription_openai_client = Client(
        task=Task.TRANSCRIPTION, providers=[Provider.OPENAI]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    response, metadata = transcription_openai_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSCRIPTION
    assert metadata.provider == Provider.OPENAI
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)

from test_utils import print_request_response_and_metadata

from supercontrast import (
    Provider,
    SuperContrastClient,
    Task,
    TaskMetadata,
    TranscriptionRequest,
    TranscriptionResponse,
)

# constants

TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"
REFERENCE_TRANSCRIPTION = "November 10th, Wednesday, 9:00 PM. I'm standing in a dark alley. After waiting several hours, the time has come. A woman with long dark hair approaches. I have to act, and fast, before she realizes what has happened. I must find out."

# tests


def test_transcription_azure():
    transcription_azure_client = SuperContrastClient(
        task=Task.TRANSCRIPTION, providers=[Provider.AZURE]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    reference_response = TranscriptionResponse(text=REFERENCE_TRANSCRIPTION)
    response, metadata = transcription_azure_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSCRIPTION
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSCRIPTION, request, response, metadata)


def test_transcription_openai():
    transcription_openai_client = SuperContrastClient(
        task=Task.TRANSCRIPTION, providers=[Provider.OPENAI]
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    reference_response = TranscriptionResponse(text=REFERENCE_TRANSCRIPTION)
    response, metadata = transcription_openai_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSCRIPTION
    assert metadata.provider == Provider.OPENAI
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSCRIPTION, request, response, metadata)


# evaluate


def test_transcription_evaluate():
    transcription_client = SuperContrastClient(
        task=Task.TRANSCRIPTION,
        providers=[
            Provider.AZURE,
            Provider.OPENAI,
            # Add other providers here if available for transcription
        ],
    )
    request = TranscriptionRequest(audio_file=TEST_AUDIO_URL)
    reference_response = TranscriptionResponse(text=REFERENCE_TRANSCRIPTION)
    responses = transcription_client.evaluate(request, reference=reference_response)

    assert responses is not None
    assert isinstance(responses, dict)
    assert all(
        isinstance(response, tuple)
        and len(response) == 2
        and all(
            isinstance(item, (TranscriptionResponse, TaskMetadata)) for item in response
        )
        for response in responses.values()
    )

    for _, (response, metadata) in responses.items():
        print_request_response_and_metadata(
            Task.TRANSCRIPTION, request, response, metadata
        )

from supercontrast.client import supercontrast_client
from supercontrast.provider.provider_enum import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.types.transcription_types import TranscriptionRequest


def test_azure_transcription():
    input_audio = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"

    client = supercontrast_client(
        task=Task.TRANSCRIPTION,
        providers=[Provider.AZURE],
        optimizer=None,
    )

    response = client.request(TranscriptionRequest(audio_file=input_audio))
    assert response.text is not None
    

def test_openai_transcription():
    input_audio = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"

    client = supercontrast_client(
        task=Task.TRANSCRIPTION,
        providers=[Provider.OPENAI],
        optimizer=None,
    )

    response = client.request(TranscriptionRequest(audio_file=input_audio))
    assert response.text is not None
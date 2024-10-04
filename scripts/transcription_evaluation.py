import json

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, TranscriptionRequest

# Define test audio file

TEST_AUDIO_FILE = "test_data/test_audio.wav"

# Define providers

PROVIDERS = [
    Provider.AZURE,
    Provider.OPENAI,
]

# Define the evaluation function


def evaluate_transcription():
    # Initialize the client with all providers
    transcription_client = supercontrast_client(
        task=Task.TRANSCRIPTION, providers=PROVIDERS
    )

    # Evaluate all providers for the test audio file
    request = TranscriptionRequest(audio_file=TEST_AUDIO_FILE)
    responses = transcription_client.evaluate(request)

    result = {"input": TEST_AUDIO_FILE, "outputs": {}}

    for provider, response in responses.items():
        result["outputs"][provider.value] = {"text": response.text}

    # Save results to a JSON file
    with open("test_data/transcription_evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print the JSON results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    evaluate_transcription()

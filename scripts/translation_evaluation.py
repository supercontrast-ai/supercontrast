import json
import os
import subprocess

from typing import Optional

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, TranslationRequest

# Define test texts

TEST_TEXTS = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "I love programming in Python!",
    "Artificial intelligence is changing the world.",
    "Please translate this sentence to the target language.",
]

# Define providers

PROVIDERS = [
    Provider.ANTHROPIC,
    Provider.AWS,
    Provider.AZURE,
    Provider.GCP,
    Provider.OPENAI,
]

# Define source and target languages

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = (
    "es"  # Spanish, but you can change this to any desired target language
)

# Helper functions


def convert_to_mp3(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert the input audio file to MP3 format using FFmpeg.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str, optional): Path to the output MP3 file. If not provided,
                                     it will be generated based on the input file name.

    Returns:
        str: Path to the converted MP3 file.
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".mp3"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-acodec",
                "libmp3lame",
                "-b:a",
                "128k",
                output_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file to MP3: {e.stderr}")
        return input_file


# Define the evaluation function


def evaluate_translation():
    # Initialize the client with all providers

    translation_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=PROVIDERS,
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
    )

    # Evaluate all providers for each test text

    results = []

    for text in TEST_TEXTS:
        request = TranslationRequest(text=text)
        responses = translation_client.evaluate(request)

        result = {"input": text, "outputs": {}}

        for provider, response in responses.items():
            result["outputs"][provider.value] = {"translated_text": response.text}

        results.append(result)

    # Save results to a JSON file

    with open("test_data/translation_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print the JSON results

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_translation()

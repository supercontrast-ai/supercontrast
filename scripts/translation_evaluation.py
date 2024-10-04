import json

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

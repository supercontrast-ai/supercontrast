import json

from supercontrast import (
    Provider,
    SuperContrastClient,
    Task,
    TaskMetadata,
    TranslationRequest,
    TranslationResponse,
)

# Define test texts and providers

TEST_TEXTS = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "I love programming in Python!",
    "Artificial intelligence is changing the world.",
    "Please translate this sentence to the target language.",
]

PROVIDERS = [
    Provider.ANTHROPIC,
    Provider.AWS,
    Provider.AZURE,
    Provider.GCP,
    Provider.OPENAI,
    Provider.MODERNMT,
]

# Define source and target languages

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = (
    "es"  # Spanish, but you can change this to any desired target language
)

# Helper function


def print_request_response_and_metadata(
    request: TranslationRequest, response: TranslationResponse, metadata: TaskMetadata
):
    print("\n", "-" * 80, "\n")
    print("Translation Request:")
    print(request, "\n")
    print(f"Translation Response from {metadata.provider}:")
    print(response, "\n")
    print("Metadata:")
    print(metadata, "\n")
    print("-" * 80, "\n")


# Define the evaluation function


def evaluate_translation():
    # Initialize the client with all providers
    translation_client = SuperContrastClient(
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

        for provider, (response, metadata) in responses.items():
            result["outputs"][provider.value] = {
                "translated_text": response.text,
                "latency": metadata.latency,
            }
            print_request_response_and_metadata(request, response, metadata)

        results.append(result)

    # Save results to a JSON file
    with open("test_data/translation_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print the JSON results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_translation()

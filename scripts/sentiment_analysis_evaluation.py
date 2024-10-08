import json

from supercontrast import (
    Provider,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SuperContrastClient,
    Task,
    TaskMetadata,
)

# Define test texts
TEST_TEXTS = [
    "I love programming in Python!",
    "This movie was terrible and a waste of time.",
    "The weather today is just okay, nothing special.",
    "I'm feeling excited about the upcoming vacation!",
    "The customer service was unhelpful and frustrating.",
]

# Define providers
PROVIDERS = [
    Provider.ANTHROPIC,
    Provider.AWS,
    Provider.AZURE,
    Provider.GCP,
    Provider.OPENAI,
]


# Helper function
def print_request_response_and_metadata(
    request: SentimentAnalysisRequest,
    response: SentimentAnalysisResponse,
    metadata: TaskMetadata,
):
    print("\n", "-" * 80, "\n")
    print("Sentiment Analysis Request:")
    print(request, "\n")
    print(f"Sentiment Analysis Response from {metadata.provider}:")
    print(response, "\n")
    print("Metadata:")
    print(metadata, "\n")
    print("-" * 80, "\n")


# Define the evaluation function
def evaluate_sentiment_analysis():
    # Initialize the client with all providers
    sentiment_analysis_client = SuperContrastClient(
        task=Task.SENTIMENT_ANALYSIS, providers=PROVIDERS
    )

    results = []

    for text in TEST_TEXTS:
        request = SentimentAnalysisRequest(text=text)
        responses = sentiment_analysis_client.evaluate(request)

        result = {"input": text, "outputs": {}}

        for provider, (response, metadata) in responses.items():
            result["outputs"][provider.value] = {
                "score": response.score,
                "latency": metadata.latency,
            }
            print_request_response_and_metadata(request, response, metadata)

        results.append(result)

    # Save results to a JSON file
    with open("test_data/sentiment_analysis_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print the JSON results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_sentiment_analysis()

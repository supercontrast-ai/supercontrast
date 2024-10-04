import json

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import SentimentAnalysisRequest, Task

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


def evaluate_sentiment_analysis():
    # Initialize the client with all providers

    sentiment_analysis_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=PROVIDERS
    )

    # Evaluate all providers for each test text

    results = []

    for text in TEST_TEXTS:
        request = SentimentAnalysisRequest(text=text)
        responses = sentiment_analysis_client.evaluate(request)

        result = {"input": text, "outputs": {}}

        for provider, response in responses.items():
            result["outputs"][provider.value] = {"score": response.score}

        results.append(result)

    # Save results to a JSON file

    with open("test_data/sentiment_analysis_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print the JSON results

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_sentiment_analysis()

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import SentimentAnalysisRequest, SentimentAnalysisResponse, Task

# helper functions


def print_request_and_response(
    request: SentimentAnalysisRequest,
    response: SentimentAnalysisResponse,
    provider: Provider,
):
    print("\n", "-" * 80, "\n")
    print("Sentiment Analysis Request:")
    print(request, "\n")
    print(f"Sentiment Analysis Response from {provider}:")
    print(response, "\n")
    print("-" * 80, "\n")


# tests

TEST_TEXT = "I love programming in Python!"


def test_sentiment_analysis_anthropic():
    sentiment_analysis_anthropic_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.ANTHROPIC]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response = sentiment_analysis_anthropic_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.ANTHROPIC)


def test_sentiment_analysis_aws():
    sentiment_analysis_aws_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response = sentiment_analysis_aws_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.AWS)


def test_sentiment_analysis_azure():
    sentiment_analysis_azure_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AZURE]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response = sentiment_analysis_azure_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.AZURE)


def test_sentiment_analysis_gcp():
    sentiment_analysis_gcp_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.GCP]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response = sentiment_analysis_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.GCP)


def test_sentiment_analysis_openai():
    sentiment_analysis_openai_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.OPENAI]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response = sentiment_analysis_openai_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.OPENAI)

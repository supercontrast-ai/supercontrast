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


def test_sentiment_analysis_aws():
    sentiment_analysis_aws_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS]
    )
    request = SentimentAnalysisRequest(text="I love programming in Python!")
    response = sentiment_analysis_aws_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.AWS)


def test_sentiment_analysis_azure():
    sentiment_analysis_azure_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AZURE]
    )
    request = SentimentAnalysisRequest(text="I love programming in Python!")
    response = sentiment_analysis_azure_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.AZURE)


def test_sentiment_analysis_gcp():
    sentiment_analysis_gcp_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.GCP]
    )
    request = SentimentAnalysisRequest(text="I love programming in Python!")
    response = sentiment_analysis_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    print_request_and_response(request, response, provider=Provider.GCP)

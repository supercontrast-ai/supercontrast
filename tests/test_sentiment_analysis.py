from supercontrast import (
    Client,
    Provider,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    Task,
    TaskMetadata,
)

# constants

TEST_TEXT = "I love programming in Python!"

# helper functions


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


# tests


def test_sentiment_analysis_anthropic():
    sentiment_analysis_anthropic_client = Client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.ANTHROPIC]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response, metadata = sentiment_analysis_anthropic_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.SENTIMENT_ANALYSIS
    assert metadata.provider == Provider.ANTHROPIC
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_sentiment_analysis_aws():
    sentiment_analysis_aws_client = Client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response, metadata = sentiment_analysis_aws_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.SENTIMENT_ANALYSIS
    assert metadata.provider == Provider.AWS
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_sentiment_analysis_azure():
    sentiment_analysis_azure_client = Client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AZURE]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response, metadata = sentiment_analysis_azure_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.SENTIMENT_ANALYSIS
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_sentiment_analysis_gcp():
    sentiment_analysis_gcp_client = Client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.GCP]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response, metadata = sentiment_analysis_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.SENTIMENT_ANALYSIS
    assert metadata.provider == Provider.GCP
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_sentiment_analysis_openai():
    sentiment_analysis_openai_client = Client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.OPENAI]
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    response, metadata = sentiment_analysis_openai_client.request(request)

    assert response is not None
    assert isinstance(response.score, float)
    assert response.score > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.SENTIMENT_ANALYSIS
    assert metadata.provider == Provider.OPENAI
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)

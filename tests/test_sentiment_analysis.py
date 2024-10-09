from supercontrast import (
    Provider,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SuperContrastClient,
    Task,
    TaskMetadata,
)
from test_utils import print_request_response_and_metadata

# constants

TEST_TEXT = "I love programming in Python!"


# tests


def test_sentiment_analysis_anthropic():
    sentiment_analysis_anthropic_client = SuperContrastClient(
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

    print_request_response_and_metadata(
        Task.SENTIMENT_ANALYSIS, request, response, metadata
    )


def test_sentiment_analysis_aws():
    sentiment_analysis_aws_client = SuperContrastClient(
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

    print_request_response_and_metadata(
        Task.SENTIMENT_ANALYSIS, request, response, metadata
    )


def test_sentiment_analysis_azure():
    sentiment_analysis_azure_client = SuperContrastClient(
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

    print_request_response_and_metadata(
        Task.SENTIMENT_ANALYSIS, request, response, metadata
    )


def test_sentiment_analysis_gcp():
    sentiment_analysis_gcp_client = SuperContrastClient(
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

    print_request_response_and_metadata(
        Task.SENTIMENT_ANALYSIS, request, response, metadata
    )


def test_sentiment_analysis_openai():
    sentiment_analysis_openai_client = SuperContrastClient(
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

    print_request_response_and_metadata(
        Task.SENTIMENT_ANALYSIS, request, response, metadata
    )


# evaluate


def test_sentiment_analysis_evaluate():
    sentiment_analysis_client = SuperContrastClient(
        task=Task.SENTIMENT_ANALYSIS,
        providers=[
            Provider.ANTHROPIC,
            Provider.AWS,
            Provider.AZURE,
            Provider.GCP,
            Provider.OPENAI,
        ],
    )
    request = SentimentAnalysisRequest(text=TEST_TEXT)
    responses = sentiment_analysis_client.evaluate(request)

    assert responses is not None
    assert isinstance(responses, dict)
    assert all(
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], SentimentAnalysisResponse)
        and isinstance(response[1], TaskMetadata)
        for response in responses.values()
    )

    for provider, (response, metadata) in responses.items():
        print_request_response_and_metadata(
            Task.SENTIMENT_ANALYSIS, request, response, metadata
        )

        assert isinstance(response.score, float)
        assert response.score > 0

        assert metadata.task == Task.SENTIMENT_ANALYSIS
        assert metadata.provider == provider
        assert metadata.latency > 0

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import SentimentAnalysisRequest, Task
from supercontrast.task.handlers.sentiment_analysis_handler import (
    SentimentAnalysisHandler,
)

# helper functions


def print_request_and_response(request, response):
    print("-" * 80, "\n")
    print("Sentiment Analysis Request:")
    print(request, "\n")
    print("Sentiment Analysis Response from AWS:")
    print(response, "\n")
    print("-" * 80)


# tests


def test_sentiment_analysis_aws():
    sentiment_analysis_aws_client = supercontrast_client(
        task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS]
    )
    request = SentimentAnalysisRequest(text="I love programming in Python!")
    response = sentiment_analysis_aws_client.request(request)

    assert response != None
    print_request_and_response(request, response)

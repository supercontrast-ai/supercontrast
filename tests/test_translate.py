from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, TranslationRequest

# helper functions


def print_request_and_response(request, response):
    print("-" * 80, "\n")
    print("Translation Request:")
    print(request, "\n")
    print("Translation Response:")
    print(response, "\n")
    print("-" * 80)


# tests


def test_translate_aws():
    translate_aws_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.AWS],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text="Hello, world!")
    response = translate_aws_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text
    print_request_and_response(request, response)


def test_translate_azure():
    translate_azure_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.AZURE],
        source_language="en",
        target_language="fr",
    )
    request = TranslationRequest(text="Good morning!")
    response = translate_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text
    print_request_and_response(request, response)


def test_translate_gcp():
    translate_gcp_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.GCP],
        source_language="en",
        target_language="de",
    )
    request = TranslationRequest(text="How are you?")
    response = translate_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text
    print_request_and_response(request, response)

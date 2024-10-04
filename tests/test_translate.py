from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, TranslationRequest, TranslationResponse

# helper functions


def print_request_and_response(
    request: TranslationRequest, response: TranslationResponse, provider: Provider
):
    print("\n", "-" * 80, "\n")
    print("Translation Request:")
    print(request, "\n")
    print(f"Translation Response from {provider}:")
    print(response, "\n")
    print("-" * 80, "\n")


# tests

TEST_TEXT = "Hello, world! This is a test translation."


def test_translate_anthropic():
    translate_anthropic_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.ANTHROPIC],
        source_language="en",
        target_language="ja",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_anthropic_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.ANTHROPIC)


def test_translate_aws():
    translate_aws_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.AWS],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_aws_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.AWS)


def test_translate_azure():
    translate_azure_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.AZURE],
        source_language="en",
        target_language="fr",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.AZURE)


def test_translate_gcp():
    translate_gcp_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.GCP],
        source_language="en",
        target_language="de",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.GCP)


def test_translate_modernmt():
    translate_modernmt_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.MODERNMT],
        source_language="en",
        target_language="it",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_modernmt_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.MODERNMT)


def test_translate_openai():
    translate_openai_client = supercontrast_client(
        task=Task.TRANSLATION,
        providers=[Provider.OPENAI],
        source_language="en",
        target_language="zh",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response = translate_openai_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    print_request_and_response(request, response, provider=Provider.OPENAI)

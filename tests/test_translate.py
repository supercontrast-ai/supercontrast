from supercontrast import (
    Client,
    Provider,
    Task,
    TaskMetadata,
    TranslationRequest,
    TranslationResponse,
)

# constants

TEST_TEXT = "Hello, world! This is a test translation."

# helper functions


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


# tests


def test_translate_anthropic():
    translate_anthropic_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.ANTHROPIC],
        source_language="en",
        target_language="ja",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_anthropic_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.ANTHROPIC
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_translate_aws():
    translate_aws_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.AWS],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_aws_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.AWS
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_translate_azure():
    translate_azure_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.AZURE],
        source_language="en",
        target_language="fr",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_translate_gcp():
    translate_gcp_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.GCP],
        source_language="en",
        target_language="de",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.GCP
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_translate_modernmt():
    translate_modernmt_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.MODERNMT],
        source_language="en",
        target_language="it",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_modernmt_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.MODERNMT
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_translate_openai():
    translate_openai_client = Client(
        task=Task.TRANSLATION,
        providers=[Provider.OPENAI],
        source_language="en",
        target_language="zh",
    )
    request = TranslationRequest(text=TEST_TEXT)
    response, metadata = translate_openai_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.OPENAI
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)

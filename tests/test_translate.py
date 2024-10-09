from supercontrast import (
    Provider,
    SuperContrastClient,
    Task,
    TaskMetadata,
    TranslationRequest,
    TranslationResponse,
)
from test_utils import print_request_response_and_metadata

# constants

TEST_TEXT = "Hello, world! This is a test translation."
REFERENCE_TRANSLATIONS = {
    "it": "Ciao, mondo! Questo è un test di traduzione.",
    "fr": "Bonjour, le monde! Ceci est une traduction de test.",
    "de": "Hallo, Welt! Dies ist ein Test der Übersetzung.",
    "es": "Hola, mundo! Esto es una prueba de traducción.",
}

# tests


def test_translate_anthropic():
    translate_anthropic_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.ANTHROPIC],
        source_language="en",
        target_language="it",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["it"])
    response, metadata = translate_anthropic_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.ANTHROPIC
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


def test_translate_aws():
    translate_aws_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.AWS],
        source_language="en",
        target_language="fr",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["fr"])
    response, metadata = translate_aws_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.AWS
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


def test_translate_azure():
    translate_azure_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.AZURE],
        source_language="en",
        target_language="de",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["de"])
    response, metadata = translate_azure_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


def test_translate_gcp():
    translate_gcp_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.GCP],
        source_language="en",
        target_language="it",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["it"])
    response, metadata = translate_gcp_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.GCP
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


def test_translate_modernmt():
    translate_modernmt_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.MODERNMT],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["es"])
    response, metadata = translate_modernmt_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.MODERNMT
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


def test_translate_openai():
    translate_openai_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[Provider.OPENAI],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["es"])
    response, metadata = translate_openai_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text != request.text

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.TRANSLATION
    assert metadata.provider == Provider.OPENAI
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.TRANSLATION, request, response, metadata)


# evaluate


def test_translate_evaluate():
    translate_client = SuperContrastClient(
        task=Task.TRANSLATION,
        providers=[
            Provider.ANTHROPIC,
            Provider.MODERNMT,
            Provider.OPENAI,
            Provider.AZURE,
            Provider.GCP,
            Provider.AWS,
        ],
        source_language="en",
        target_language="es",
    )
    request = TranslationRequest(text=TEST_TEXT)
    reference_response = TranslationResponse(text=REFERENCE_TRANSLATIONS["es"])
    responses = translate_client.evaluate(request, reference=reference_response)

    assert responses is not None
    assert isinstance(responses, dict)
    assert all(
        isinstance(response, tuple)
        and len(response) == 2
        and all(
            isinstance(item, (TranslationResponse, TaskMetadata)) for item in response
        )
        for response in responses.values()
    )

    for _, (response, metadata) in responses.items():
        print_request_response_and_metadata(
            Task.TRANSLATION, request, response, metadata
        )

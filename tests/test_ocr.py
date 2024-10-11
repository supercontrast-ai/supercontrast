from test_utils import print_request_response_and_metadata

from supercontrast import (
    OCRRequest,
    OCRResponse,
    Provider,
    SuperContrastClient,
    Task,
    TaskMetadata,
    get_supported_providers_for_task,
)

# constants

TEST_IMAGE_URL = "https://github.com/supercontrast-ai/supercontrast/raw/main/tests/image/test_ocr.png"
REFERENCE_OCR_TEXT = "This is a lot of 12 point text to test the\nocr code and see if it works on all types\nof file format.\nThe quick brown dog jumped over the\nlazy fox. The quick brown dog jumped\nover the lazy fox. The quick brown dog\njumped over the lazy fox. The quick\nbrown dog jumped over the lazy fox."


# tests


def test_ocr_api4ai():
    ocr_api4ai_client = SuperContrastClient(task=Task.OCR, providers=[Provider.API4AI])
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_api4ai_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.API4AI
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


def test_ocr_aws():
    ocr_aws_client = SuperContrastClient(task=Task.OCR, providers=[Provider.AWS])
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_aws_client.request(request, reference=reference_response)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.AWS
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


def test_ocr_azure():
    ocr_azure_client = SuperContrastClient(task=Task.OCR, providers=[Provider.AZURE])
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_azure_client.request(request, reference=reference_response)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


def test_ocr_clarifai():
    ocr_clarifai_client = SuperContrastClient(
        task=Task.OCR, providers=[Provider.CLARIFAI]
    )
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_clarifai_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.CLARIFAI
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


def test_ocr_gcp():
    ocr_gcp_client = SuperContrastClient(task=Task.OCR, providers=[Provider.GCP])
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_gcp_client.request(request, reference=reference_response)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.GCP
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


def test_ocr_sentisight():
    ocr_sentisight_client = SuperContrastClient(
        task=Task.OCR, providers=[Provider.SENTISIGHT]
    )
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    response, metadata = ocr_sentisight_client.request(
        request, reference=reference_response
    )

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.SENTISIGHT
    assert metadata.latency > 0

    print_request_response_and_metadata(Task.OCR, request, response, metadata)


# evaluate


def test_ocr_evaluate():
    ocr_client = SuperContrastClient(
        task=Task.OCR, providers=get_supported_providers_for_task(Task.OCR)
    )
    request = OCRRequest(image=TEST_IMAGE_URL)
    reference_response = OCRResponse(all_text=REFERENCE_OCR_TEXT, bounding_boxes=[])
    responses = ocr_client.evaluate(request, reference=reference_response)

    assert responses is not None
    assert isinstance(responses, dict)
    assert all(
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], OCRResponse)
        and isinstance(response[1], TaskMetadata)
        for response in responses.values()
    )

    for provider, (response, metadata) in responses.items():
        print_request_response_and_metadata(Task.OCR, request, response, metadata)

        assert isinstance(response.all_text, str)
        assert len(response.all_text) > 0
        assert len(response.bounding_boxes) > 0

        assert metadata.task == Task.OCR
        assert metadata.provider == provider
        assert metadata.latency > 0

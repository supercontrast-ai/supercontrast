from supercontrast import Client, OCRRequest, OCRResponse, Provider, Task, TaskMetadata

# constants

TEST_IMAGE_URL = "https://jeroen.github.io/images/testocr.png"

# helper functions


def print_request_response_and_metadata(
    request: OCRRequest, response: OCRResponse, metadata: TaskMetadata
):
    print("\n", "-" * 80, "\n")
    print("OCR Request:")
    print(request, "\n")
    print(f"OCR Response from {metadata.provider}:")
    print(response, "\n")
    print("Metadata:")
    print(metadata, "\n")
    print("-" * 80, "\n")


# tests


def test_ocr_api4ai():
    ocr_api4ai_client = Client(task=Task.OCR, providers=[Provider.API4AI])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_api4ai_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.API4AI
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_ocr_aws():
    ocr_aws_client = Client(task=Task.OCR, providers=[Provider.AWS])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_aws_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.AWS
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_ocr_azure():
    ocr_azure_client = Client(task=Task.OCR, providers=[Provider.AZURE])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_azure_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.AZURE
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_ocr_clarifai():
    ocr_clarifai_client = Client(task=Task.OCR, providers=[Provider.CLARIFAI])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_clarifai_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.CLARIFAI
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_ocr_gcp():
    ocr_gcp_client = Client(task=Task.OCR, providers=[Provider.GCP])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.GCP
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)


def test_ocr_sentisight():
    ocr_sentisight_client = Client(task=Task.OCR, providers=[Provider.SENTISIGHT])
    request = OCRRequest(image=TEST_IMAGE_URL)
    response, metadata = ocr_sentisight_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    assert metadata is not None
    assert isinstance(metadata, TaskMetadata)
    assert metadata.task == Task.OCR
    assert metadata.provider == Provider.SENTISIGHT
    assert metadata.latency > 0

    print_request_response_and_metadata(request, response, metadata)

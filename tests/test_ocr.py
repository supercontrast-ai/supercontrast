from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import OCRRequest, OCRResponse, Task

# helper functions


def print_request_and_response(
    request: OCRRequest, response: OCRResponse, provider: Provider
):
    print("\n", "-" * 80, "\n")
    print("OCR Request:")
    print(request, "\n")
    print(f"OCR Response from {provider}:")
    print(response, "\n")
    print("-" * 80, "\n")


# tests


def test_ocr_aws():
    ocr_aws_client = supercontrast_client(task=Task.OCR, providers=[Provider.AWS])
    request = OCRRequest(image="https://jeroen.github.io/images/testocr.png")
    response = ocr_aws_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    print_request_and_response(request, response, provider=Provider.AWS)


def test_ocr_azure():
    ocr_azure_client = supercontrast_client(task=Task.OCR, providers=[Provider.AZURE])
    request = OCRRequest(image="https://jeroen.github.io/images/testocr.png")
    response = ocr_azure_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    print_request_and_response(request, response, provider=Provider.AZURE)


def test_ocr_gcp():
    ocr_gcp_client = supercontrast_client(task=Task.OCR, providers=[Provider.GCP])
    request = OCRRequest(image="https://jeroen.github.io/images/testocr.png")
    response = ocr_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    print_request_and_response(request, response, provider=Provider.GCP)


def test_ocr_sentisight():
    ocr_sentisight_client = supercontrast_client(
        task=Task.OCR, providers=[Provider.SENTISIGHT]
    )
    request = OCRRequest(image="https://jeroen.github.io/images/testocr.png")
    response = ocr_sentisight_client.request(request)

    assert response is not None
    assert isinstance(response.all_text, str)
    assert len(response.all_text) > 0
    assert len(response.bounding_boxes) > 0

    print_request_and_response(request, response, provider=Provider.SENTISIGHT)

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import OCRRequest, Task

# helper functions


def print_request_and_response(request, response):
    print("-" * 80, "\n")
    print("OCR Request:")
    print(request, "\n")
    print("OCR Response from AWS:")
    print(response, "\n")
    print("-" * 80)


# tests


def test_ocr_aws():
    ocr_aws_client = supercontrast_client(task=Task.OCR, providers=[Provider.AWS])
    request = OCRRequest(
        image="./test_data/ocr_sample.png"
    )  # Replace with actual test image path
    response = ocr_aws_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print_request_and_response(request, response)


def test_ocr_azure():
    ocr_azure_client = supercontrast_client(task=Task.OCR, providers=[Provider.AZURE])
    request = OCRRequest(
        image="./test_data/ocr_sample.png"
    )  # Replace with actual test image path
    response = ocr_azure_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print_request_and_response(request, response)


def test_ocr_gcp():
    ocr_gcp_client = supercontrast_client(task=Task.OCR, providers=[Provider.GCP])
    request = OCRRequest(
        image="./test_data/ocr_sample.png"
    )  # Replace with actual test image path
    response = ocr_gcp_client.request(request)

    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print_request_and_response(request, response)

import io
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import requests

from PIL import Image

from supercontrast import (
    OCRRequest,
    OCRResponse,
    Provider,
    SuperContrastClient,
    Task,
    TaskMetadata,
)

# Constants
TEST_IMAGE_URLS = [
    "https://jeroen.github.io/images/testocr.png",
    # Add more image URLs here
]

PROVIDERS = [
    Provider.API4AI,
    Provider.AWS,
    Provider.AZURE,
    Provider.CLARIFAI,
    Provider.GCP,
    Provider.SENTISIGHT,
]

OUTPUT_DIR = "test_data/ocr"

# Helper functions


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


def plot_bounding_boxes(
    image_url: str, response: OCRResponse, provider: Provider, output_dir: str
):
    # Download the image
    response_img = requests.get(image_url)
    img = Image.open(io.BytesIO(response_img.content))

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for box in response.bounding_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (box.coordinates[0][0], box.coordinates[0][1]),
            box.coordinates[2][0] - box.coordinates[0][0],
            box.coordinates[2][1] - box.coordinates[0][1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the plot as an image file
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_url)
    output_path = os.path.join(output_dir, f"ocr_{provider.value}_{image_name}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved {provider.value} plot to: {output_path}")

    plt.close(fig)


# Evaluation function


def evaluate_ocr():
    ocr_client = SuperContrastClient(task=Task.OCR, providers=PROVIDERS)

    results = []

    for image_url in TEST_IMAGE_URLS:
        print(f"Evaluating image: {image_url}")
        request = OCRRequest(image=image_url)
        responses = ocr_client.evaluate(request)

        result = {"input": image_url, "outputs": {}}

        for provider, (response, metadata) in responses.items():
            result["outputs"][provider.value] = {
                "text": response.all_text,
                "latency": metadata.latency,
            }
            print_request_response_and_metadata(request, response, metadata)
            plot_bounding_boxes(image_url, response, provider, OUTPUT_DIR)

        results.append(result)

    # Save results to a JSON file
    with open(os.path.join(OUTPUT_DIR, "ocr_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print the JSON results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_ocr()

import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import requests

from PIL import Image

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import OCRRequest, OCRResponse, Task

# Define providers

PROVIDERS = [
    Provider.AWS,
    Provider.GCP,
    Provider.AZURE,
    Provider.SENTISIGHT,
    Provider.CLARIFAI,
    Provider.API4AI,
    # Provider.OPENAI,
]

# Define image URLs to evaluate

IMAGE_URLS = [
    "https://jeroen.github.io/images/testocr.png",
    # Add more image URLs here
]

# Define output directory for saving plots

OUTPUT_DIR = "test_data/ocr"


def plot_bounding_boxes(
    image_url: str, responses: dict[Provider, OCRResponse], output_dir: str
):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    for provider, ocr_response in responses.items():
        # Create a new figure for each provider
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for box in ocr_response.bounding_boxes:
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

        # Remove any extra white space around the image
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot as an image file
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_url)
        output_path = os.path.join(output_dir, f"ocr_{provider.value}_{image_name}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved {provider.value} plot to: {output_path}")

        plt.close(fig)


def main():
    # Initialize the OCR client with all providers
    ocr_client = supercontrast_client(
        task=Task.OCR,
        providers=PROVIDERS,
    )

    for image_url in IMAGE_URLS:
        print(f"Evaluating image: {image_url}")

        # Create OCR request
        request = OCRRequest(image=image_url)

        # Evaluate all providers
        responses = ocr_client.evaluate(request)

        # Plot bounding boxes and save individual plots
        plot_bounding_boxes(image_url, responses, OUTPUT_DIR)

        # Print text results
        for provider, response in responses.items():
            print(f"\nProvider: {provider.value}")
            print(f"Detected text: {response.all_text}")


if __name__ == "__main__":
    main()

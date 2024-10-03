import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import requests

from PIL import Image

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import OCRRequest, OCRResponse, Task


def plot_bounding_boxes(
    image_url: str, responses: dict[Provider, OCRResponse], output_dir: str
):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    # Create a figure with subplots for each provider
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("OCR Bounding Boxes Comparison", fontsize=16)

    for idx, (provider, ocr_response) in enumerate(responses.items()):
        ax = axs[idx // 2, idx % 2]
        ax.imshow(img)
        ax.set_title(f"Provider: {provider.value}")

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

    plt.tight_layout()

    # Save the plot as an image file
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_url)
    output_path = os.path.join(output_dir, f"ocr_comparison_{image_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {output_path}")

    plt.show()


def main():
    # Initialize the OCR client with all providers
    ocr_client = supercontrast_client(
        task=Task.OCR,
        providers=[Provider.AWS, Provider.GCP, Provider.AZURE, Provider.SENTISIGHT],
    )

    # List of image URLs to evaluate
    image_urls = [
        "https://jeroen.github.io/images/testocr.png",
        # Add more image URLs here
    ]

    # Output directory for saving plots
    output_dir = "test_data"

    for image_url in image_urls:
        print(f"Evaluating image: {image_url}")

        # Create OCR request
        request = OCRRequest(image=image_url)

        # Evaluate all providers
        responses = ocr_client.evaluate(request)

        # Plot bounding boxes and save the plot
        plot_bounding_boxes(image_url, responses, output_dir)

        # Print text results
        for provider, response in responses.items():
            print(f"\nProvider: {provider.value}")
            print(f"Detected text: {response.all_text}")


if __name__ == "__main__":
    main()

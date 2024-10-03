import requests

from io import BytesIO
from PIL import Image
from typing import Tuple, Union


def get_image_size(image: Union[str, bytes]) -> Tuple[int, int]:
    """
    Get the size of an image.

    Args:
        image (Union[str, bytes]): The image to get the size of. Can be a URL, a local file path, or bytes.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.

    Raises:
        ValueError: If the image type is unsupported.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            response = requests.get(image)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(BytesIO(image))
    else:
        raise ValueError("Unsupported image type")

    return img.size


def load_image_data(image: Union[str, bytes]) -> bytes:
    """
    Load image data from various sources.

    Args:
        image (Union[str, bytes]): The image to load. Can be a URL, a local file path, or bytes.

    Returns:
        bytes: The image data as bytes.

    Raises:
        ValueError: If the image type is unsupported.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            return requests.get(image).content
        else:
            with open(image, "rb") as image_file:
                return image_file.read()
    elif isinstance(image, bytes):
        return image
    else:
        raise ValueError("Unsupported image type")

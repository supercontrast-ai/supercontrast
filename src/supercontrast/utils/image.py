import base64
import os
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


# Image processing for LLMs


def encode_image(image_data: bytes) -> str:
    return base64.b64encode(image_data).decode("utf-8")


def convert_to_jpeg_and_resize(image_data: bytes, max_size: int = 1024) -> bytes | None:
    try:
        with Image.open(BytesIO(image_data)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            original_width, original_height = img.size
            ratio = min(max_size / original_width, max_size / original_height)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            img = img.resize(new_size)
            output_buffer = BytesIO()
            img.save(output_buffer, format="JPEG")
            return output_buffer.getvalue()
    except IOError:
        print("Error in converting the image to JPEG")
        return None


def process_image_for_llm(image_data: bytes) -> str | None:
    jpeg_data = convert_to_jpeg_and_resize(image_data)
    if jpeg_data is None:
        return None
    return encode_image(jpeg_data)

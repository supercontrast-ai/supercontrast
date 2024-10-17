from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.ocr_types import OCRRequest, OCRResponse

from pyzerox import zerox

from supercontrast.utils.image import load_image_data, process_image_for_llm
import asyncio


class OmniOCR(ProviderHandler):
    def __init__(self):
        super().__init__(provider=Provider.OMNI, task=Task.OCR)

    def request(self, request: OCRRequest) -> OCRResponse:
        pdf_data = load_image_data(request.image)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(zerox(pdf_data))


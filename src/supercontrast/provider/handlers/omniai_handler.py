import asyncio
import os

from pyzerox import zerox
from typing import Optional

from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task
from supercontrast.task.types.document_reconstruction_types import (
    DocumentReconstructionRequest,
    DocumentReconstructionResponse,
)

# Constants
OMNIAI_SUPPORTED_TASKS = [Task.DOCUMENT_RECONSTRUCTION]

# Task.DOCUMENT_RECONSTRUCTION


class OmniAIDocumentReconstruction(ProviderHandler):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(provider=Provider.OMNIAI, task=Task.DOCUMENT_RECONSTRUCTION)
        self.model = model
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OPENAI_API_KEY must be set in the environment or provided"
            )

    async def async_request(
        self, request: DocumentReconstructionRequest
    ) -> DocumentReconstructionResponse:
        result = await zerox(
            file_path=request.input_file,
            model=self.model,
            output_dir=None,
            custom_system_prompt=None,
            select_pages=None,
            cleanup=False,
        )
        return DocumentReconstructionResponse(output=result)

    def request(
        self, request: DocumentReconstructionRequest
    ) -> DocumentReconstructionResponse:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_request(request))

    def get_name(self) -> str:
        return f"OmniAI Document Reconstruction ({self.model})"

    @classmethod
    def init_from_env(
        cls, model: Optional[str] = None, api_key: Optional[str] = None
    ) -> "OmniAIDocumentReconstruction":
        model = model or "gpt-4o-mini"
        return cls(model, api_key)


# factory


def omniai_provider_factory(task: Task, **config) -> ProviderHandler:
    model = config.get("model", "gpt-4o-mini")
    api_key = config.get("api_key")

    if task not in OMNIAI_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.DOCUMENT_RECONSTRUCTION:
        return OmniAIDocumentReconstruction.init_from_env(model, api_key)
    else:
        raise ValueError(f"Unsupported task: {task}")

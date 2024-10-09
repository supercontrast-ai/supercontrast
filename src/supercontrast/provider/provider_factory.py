from typing import List

from supercontrast.provider.handlers.anthropic_handler import (
    ANTHROPIC_SUPPORTED_TASKS,
    anthropic_provider_factory,
)
from supercontrast.provider.handlers.api4ai_handler import (
    API4AI_SUPPORTED_TASKS,
    api4ai_provider_factory,
)
from supercontrast.provider.handlers.aws_handler import (
    AWS_SUPPORTED_TASKS,
    aws_provider_factory,
)
from supercontrast.provider.handlers.azure_handler import (
    AZURE_SUPPORTED_TASKS,
    azure_provider_factory,
)
from supercontrast.provider.handlers.clarifai_handler import (
    CLARIFAI_SUPPORTED_TASKS,
    clarifai_provider_factory,
)
from supercontrast.provider.handlers.gcp_handler import (
    GCP_SUPPORTED_TASKS,
    gcp_provider_factory,
)
from supercontrast.provider.handlers.modern_mt_handler import (
    MODERNMT_SUPPORTED_TASKS,
    modernmt_provider_factory,
)
from supercontrast.provider.handlers.openai_handler import (
    OPENAI_SUPPORTED_TASKS,
    openai_provider_factory,
)
from supercontrast.provider.handlers.sentisight_handler import (
    SENTISIGHT_SUPPORTED_TASKS,
    sentisight_provider_factory,
)
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task


def get_supported_tasks_for_provider(provider: Provider) -> List[Task]:
    if provider == Provider.ANTHROPIC:
        return ANTHROPIC_SUPPORTED_TASKS
    elif provider == Provider.API4AI:
        return API4AI_SUPPORTED_TASKS
    elif provider == Provider.AWS:
        return AWS_SUPPORTED_TASKS
    elif provider == Provider.AZURE:
        return AZURE_SUPPORTED_TASKS
    elif provider == Provider.CLARIFAI:
        return CLARIFAI_SUPPORTED_TASKS
    elif provider == Provider.GCP:
        return GCP_SUPPORTED_TASKS
    elif provider == Provider.MODERNMT:
        return MODERNMT_SUPPORTED_TASKS
    elif provider == Provider.OPENAI:
        return OPENAI_SUPPORTED_TASKS
    elif provider == Provider.SENTISIGHT:
        return SENTISIGHT_SUPPORTED_TASKS
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def provider_factory(task: Task, provider: Provider, **config) -> ProviderHandler:
    if provider == Provider.ANTHROPIC and task in ANTHROPIC_SUPPORTED_TASKS:
        return anthropic_provider_factory(task, **config)
    elif provider == Provider.API4AI and task in API4AI_SUPPORTED_TASKS:
        return api4ai_provider_factory(task, **config)
    elif provider == Provider.AWS and task in AWS_SUPPORTED_TASKS:
        return aws_provider_factory(task, **config)
    elif provider == Provider.AZURE and task in AZURE_SUPPORTED_TASKS:
        return azure_provider_factory(task, **config)
    elif provider == Provider.CLARIFAI and task in CLARIFAI_SUPPORTED_TASKS:
        return clarifai_provider_factory(task, **config)
    elif provider == Provider.GCP and task in GCP_SUPPORTED_TASKS:
        return gcp_provider_factory(task, **config)
    elif provider == Provider.MODERNMT and task in MODERNMT_SUPPORTED_TASKS:
        return modernmt_provider_factory(task, **config)
    elif provider == Provider.OPENAI and task in OPENAI_SUPPORTED_TASKS:
        return openai_provider_factory(task, **config)
    elif provider == Provider.SENTISIGHT and task in SENTISIGHT_SUPPORTED_TASKS:
        return sentisight_provider_factory(task, **config)
    else:
        raise ValueError(f"Task {task} is not supported by provider {provider}")

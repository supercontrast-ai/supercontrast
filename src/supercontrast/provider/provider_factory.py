from supercontrast.provider.handlers.api4ai_handler import api4ai_provider_factory
from supercontrast.provider.handlers.aws_handler import aws_provider_factory
from supercontrast.provider.handlers.azure_handler import azure_provider_factory
from supercontrast.provider.handlers.gcp_handler import gcp_provider_factory
from supercontrast.provider.handlers.sentisight_handler import sentisight_provider_factory
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task


def provider_factory(task: Task, provider: Provider, **config) -> ProviderHandler:
    if provider == Provider.AWS:
        return aws_provider_factory(task, **config)
    elif provider == Provider.GCP:
        return gcp_provider_factory(task, **config)
    elif provider == Provider.AZURE:
        return azure_provider_factory(task, **config)
    elif provider == Provider.SENTISIGHT:
        return sentisight_provider_factory(task, **config)
    elif provider == Provider.API4AI:
        return api4ai_provider_factory(task, **config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

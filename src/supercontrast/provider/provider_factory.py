from supercontrast.provider.cloud.aws_provider import aws_provider_factory
from supercontrast.provider.cloud.azure_provider import azure_provider_factory
from supercontrast.provider.cloud.gcp_provider import gcp_provider_factory
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_model import ProviderModel
from supercontrast.task.task_enum import Task


def provider_factory(task: Task, provider: Provider, **config) -> ProviderModel:
    if provider == Provider.AWS:
        return aws_provider_factory(task, **config)
    elif provider == Provider.GCP:
        return gcp_provider_factory(task, **config)
    elif provider == Provider.AZURE:
        return azure_provider_factory(task, **config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

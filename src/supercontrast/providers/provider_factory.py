from supercontrast.providers.cloud.aws_provider import aws_provider_factory
from supercontrast.providers.cloud.azure_provider import azure_provider_factory
from supercontrast.providers.cloud.gcp_provider import gcp_provider_factory
from supercontrast.providers.provider_enum import Provider
from supercontrast.tasks.task_enum import Task


def provider_factory(task: Task, provider: Provider, **config):
    if provider == Provider.AWS:
        return aws_provider_factory(task, **config)
    elif provider == Provider.GCP:
        return gcp_provider_factory(task, **config)
    elif provider == Provider.AZURE:
        return azure_provider_factory(task, **config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

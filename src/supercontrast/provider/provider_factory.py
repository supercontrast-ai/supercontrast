from typing import Any, Dict, List

# Import all provider handlers
from supercontrast.provider.handlers import (
    anthropic_handler,
    api4ai_handler,
    aws_handler,
    azure_handler,
    clarifai_handler,
    gcp_handler,
    modern_mt_handler,
    omniai_handler,
    openai_handler,
    sentisight_handler,
)
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_handler import ProviderHandler
from supercontrast.task.task_enum import Task

# Define a dictionary mapping providers to their supported tasks
PROVIDER_TASKS: Dict[Provider, List[Task]] = {
    Provider.ANTHROPIC: anthropic_handler.ANTHROPIC_SUPPORTED_TASKS,
    Provider.API4AI: api4ai_handler.API4AI_SUPPORTED_TASKS,
    Provider.AWS: aws_handler.AWS_SUPPORTED_TASKS,
    Provider.AZURE: azure_handler.AZURE_SUPPORTED_TASKS,
    Provider.CLARIFAI: clarifai_handler.CLARIFAI_SUPPORTED_TASKS,
    Provider.GCP: gcp_handler.GCP_SUPPORTED_TASKS,
    Provider.MODERNMT: modern_mt_handler.MODERNMT_SUPPORTED_TASKS,
    Provider.OMNIAI: omniai_handler.OMNIAI_SUPPORTED_TASKS,
    Provider.OPENAI: openai_handler.OPENAI_SUPPORTED_TASKS,
    Provider.SENTISIGHT: sentisight_handler.SENTISIGHT_SUPPORTED_TASKS,
}

# Define a dictionary mapping providers to their factory functions
PROVIDER_FACTORIES: Dict[Provider, Any] = {
    Provider.ANTHROPIC: anthropic_handler.anthropic_provider_factory,
    Provider.API4AI: api4ai_handler.api4ai_provider_factory,
    Provider.AWS: aws_handler.aws_provider_factory,
    Provider.AZURE: azure_handler.azure_provider_factory,
    Provider.CLARIFAI: clarifai_handler.clarifai_provider_factory,
    Provider.GCP: gcp_handler.gcp_provider_factory,
    Provider.MODERNMT: modern_mt_handler.modernmt_provider_factory,
    Provider.OMNIAI: omniai_handler.omniai_provider_factory,
    Provider.OPENAI: openai_handler.openai_provider_factory,
    Provider.SENTISIGHT: sentisight_handler.sentisight_provider_factory,
}


def get_supported_tasks_for_provider(provider: Provider) -> List[Task]:
    if provider not in PROVIDER_TASKS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_TASKS[provider]


def provider_factory(task: Task, provider: Provider, **config) -> ProviderHandler:
    if provider not in PROVIDER_TASKS or provider not in PROVIDER_FACTORIES:
        raise ValueError(f"Unsupported provider: {provider}")

    if task not in PROVIDER_TASKS[provider]:
        raise ValueError(f"Task {task} is not supported by provider {provider}")

    return PROVIDER_FACTORIES[provider](task, **config)

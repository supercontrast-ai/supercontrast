import time

from abc import ABC
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from supercontrast.optimizer.optimizer_enum import Optimizer
from supercontrast.optimizer.optimizer_factory import optimizer_factory
from supercontrast.provider.provider_enum import Provider
from supercontrast.provider.provider_factory import provider_factory
from supercontrast.task.task_enum import Task
from supercontrast.task.task_metadata import TaskMetadata

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class TaskHandler(ABC, Generic[RequestType, ResponseType]):
    def __init__(
        self,
        task: Task,
        providers: List[Provider],
        optimizer: Optional[Optimizer] = None,
        **config,
    ):
        self.task = task
        self.provider_handler_map = {
            provider: provider_factory(task=task, provider=provider, **config)
            for provider in providers
        }
        self.optimizer_handler = optimizer_factory(
            task=task, providers=providers, optimizer=optimizer
        )

    def request(
        self, body: RequestType, provider: Optional[Provider] = None
    ) -> Tuple[ResponseType, TaskMetadata]:
        if provider is None:
            provider = self.optimizer_handler.get_provider()

        provider_handler = self.provider_handler_map[provider]
        start_time = time.time()
        response = provider_handler.request(body)
        latency = time.time() - start_time

        metadata = TaskMetadata(
            task=self.task, provider=provider, latency=latency, metrics={}
        )

        return response, metadata

    def evaluate(
        self, body: RequestType
    ) -> Dict[Provider, Tuple[ResponseType, TaskMetadata]]:
        responses = {}
        for provider, handler in self.provider_handler_map.items():
            try:
                start_time = time.time()
                response = handler.request(body)
                latency = time.time() - start_time

                metadata = TaskMetadata(
                    task=self.task, provider=provider, latency=latency, metrics={}
                )

                responses[provider] = (response, metadata)
            except Exception as e:
                # Log the error or handle it as appropriate for your use case
                print(f"Error evaluating provider {provider}: {str(e)}")
        return responses

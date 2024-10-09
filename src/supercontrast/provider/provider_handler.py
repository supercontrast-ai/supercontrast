from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# generic types

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class ProviderHandler(ABC, Generic[RequestType, ResponseType]):
    @abstractmethod
    def __init__(self, provider, task, *args, **kwargs):
        self.provider = provider
        self.task = task

    @abstractmethod
    def request(self, request: RequestType) -> ResponseType:
        raise NotImplementedError("ProviderHandler must implement request method")

    # @abstractmethod
    # async def request_async(self, request: RequestType) -> ResponseType:
    #     raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # def batch_request(self, requests: List[RequestType]) -> List[ResponseType]:
    #     raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("ProviderHandler must implement get_name method")

    @classmethod
    @abstractmethod
    def init_from_env(cls, *args, **kwargs) -> "ProviderHandler":
        raise NotImplementedError("ProviderHandler must implement init_from_env method")

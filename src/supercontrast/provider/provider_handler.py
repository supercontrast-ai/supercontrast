from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class ProviderHandler(ABC, Generic[RequestType, ResponseType]):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def request(self, request: RequestType) -> ResponseType:
        raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # async def request_async(self, request: RequestType) -> ResponseType:
    #     raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # def batch_request(self, requests: List[RequestType]) -> List[ResponseType]:
    #     raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # def get_name(self) -> str:
    #     raise NotImplementedError("Subclass must implement abstract method")

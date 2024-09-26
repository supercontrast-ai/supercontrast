from abc import ABC, abstractmethod


class ProviderHandler(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def request(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # async def request_async(self, *args, **kwargs):
    #     raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # def batch_request(self, *args, **kwargs):
    #     raise NotImplementedError("Subclass must implement abstract method")

    # @abstractmethod
    # def get_name(self) -> str:
    #     raise NotImplementedError("Subclass must implement abstract method")

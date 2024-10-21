from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Generic, List, TypeVar

T = TypeVar("T", bound="TaskResponse")


class TaskResponse(BaseModel, ABC, Generic[T]):

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def compare(self, other: T) -> List[str]:
        pass

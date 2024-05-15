from abc import ABC, abstractmethod
from typing import Any

class DAO(ABC):
    @abstractmethod
    def create(self, entity: Any) -> Any:
        pass

    @abstractmethod
    def read(self, entity_id: int) -> Any:
        pass

    @abstractmethod
    def update(self, entity: Any) -> Any:
        pass

    @abstractmethod
    def delete(self, entity_id: int) -> Any:
        pass

    @abstractmethod
    def find_all(self) -> list:
        pass

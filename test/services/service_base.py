from abc import ABC, abstractmethod

class ServiceBase(ABC):
    @abstractmethod
    def perform_task(self):
        pass

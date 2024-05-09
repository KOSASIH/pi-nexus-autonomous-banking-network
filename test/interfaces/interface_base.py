from abc import ABC, abstractmethod

class InterfaceBase(ABC):
    @abstractmethod
    def do_something(self):
        pass

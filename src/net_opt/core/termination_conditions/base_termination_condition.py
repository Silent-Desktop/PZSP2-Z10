from abc import ABC, abstractmethod

class TerminationCondition(ABC):
    @abstractmethod
    def check(self, lowest_penalty: float, iteration_n: int) -> bool:
        """False on termination"""
        ...
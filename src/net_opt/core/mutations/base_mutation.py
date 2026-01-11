from abc import abstractmethod, ABC
from net_opt.core.population import Population

class Mutation(ABC):
    @abstractmethod
    def mutate(self, population : Population, elite_size: int) -> Population:
        ...
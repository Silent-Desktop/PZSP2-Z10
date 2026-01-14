from abc import abstractmethod, ABC
from net_opt.core.population import Population
from jaxtyping import Bool
from torch import Tensor

class Mutation(ABC):
    @abstractmethod
    def mutate(self, population : Population, elite_size: int, neigh_matrix: Bool[Tensor, "N N"]) -> Population:
        ...
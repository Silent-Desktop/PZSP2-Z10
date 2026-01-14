from abc import abstractmethod, ABC
from jaxtyping import Float
from torch import Tensor
from net_opt.core.population import Population


class Selection(ABC):
    @abstractmethod
    def get_next_generation(
        self,
        population: Population,
        penalties: Float[Tensor, "P"],
        elite_size: int,
    ) -> Population:
        ...

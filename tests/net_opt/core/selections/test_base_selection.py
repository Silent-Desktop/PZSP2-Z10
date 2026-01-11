import pytest
import torch
from net_opt.core.population import Population
from net_opt.core.selections.base_selection import Selection 

class TestSelection:
    
    class ConcreteSelection(Selection):
        def get_next_generation(self, population: Population, penalties: torch.Tensor, elite_size: int) -> Population:
            # Dummy logic
            return population


    def test_cannot_instantiate_abstract_class(self):
        """
        Ensure that trying to instantiate the abstract Selection class directly 
        raises a TypeError.
        """
        with pytest.raises(TypeError):
            Selection() # type: ignore

    def test_concrete_implementation_works(self, population_factory, sample_data):
        """
        Verify that a valid subclass can be instantiated and the method 
        can be called with the correct types.
        """
        P = sample_data["P"]
        pop = population_factory()
        penalties = torch.zeros(P)
        elite_size = 1

        selector = self.ConcreteSelection()
        
        new_pop = selector.get_next_generation(pop, penalties, elite_size)
        
        assert isinstance(new_pop, Population)
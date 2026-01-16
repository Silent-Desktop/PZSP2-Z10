import pytest
from net_opt.core.population import Population
from net_opt.core.mutations.base_mutation import Mutation


class TestMutation:
    class ConcreteMutation(Mutation):
        def mutate(
            self, population: Population, elite_size: int
        ) -> Population:
            # Dummy logic
            return population

    def test_cannot_instantiate_abstract_class(self):
        """
        Ensure that trying to instantiate the abstract Mutation class directly
        raises a TypeError.
        """
        with pytest.raises(TypeError):
            Mutation()  # type: ignore

    def test_concrete_implementation_works(
        self, population_factory, sample_data
    ):
        """
        Verify that a valid subclass can be instantiated and the method
        can be called with the correct types.
        """
        pop = population_factory()
        elite_size = 1

        mutator = self.ConcreteMutation()

        new_pop = mutator.mutate(pop, elite_size)

        assert isinstance(new_pop, Population)

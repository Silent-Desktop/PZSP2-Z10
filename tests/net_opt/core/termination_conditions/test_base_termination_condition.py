import pytest

from net_opt.core.termination_conditions.base_termination_condition import TerminationCondition

class TestTerminationCondition:

    class ConcreteTermination(TerminationCondition):
        def check(self, lowest_penalty: float, iteration_n: int) -> bool:
            # Dummy logic
            return iteration_n <= 10

    def test_cannot_instantiate_abstract_class(self):
        """
        Ensure that trying to instantiate the abstract class directly 
        raises a TypeError.
        """
        with pytest.raises(TypeError):
            TerminationCondition() # type: ignore

    def test_concrete_implementation_works(self):
        """
        Verify that a valid subclass can be instantiated and the method 
        can be called with the correct types.
        """
        condition = self.ConcreteTermination()
        
        assert condition.check(0.5, 5) is True
        assert condition.check(0.1, 11) is False
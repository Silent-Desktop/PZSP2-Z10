from pydantic import BaseModel, PrivateAttr
from typing import Optional
from net_opt.core.termination_conditions.base_termination_condition import (
    TerminationCondition,
)


class MinImprovement(TerminationCondition, BaseModel):
    # TODO: Patience
    delta: float
    _prev_score: Optional[float] = PrivateAttr(default=None)

    def check(self, lowest_penalty: float, iteration_n: int) -> bool:
        if not self._prev_score:
            self._prev_score = lowest_penalty
            return True
        diff = abs(lowest_penalty - self._prev_score)
        self._prev_score = lowest_penalty

        return diff > self.delta

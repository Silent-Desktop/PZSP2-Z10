import time
from pydantic import BaseModel

from net_opt.core.termination_conditions.base_termination_condition import (
    TerminationCondition,
)


class TimeLimit(TerminationCondition, BaseModel):
    time_limit_s: float

    def check(self, lowest_penalty: float, iteration_n: int) -> bool:
        if not iteration_n:
            self._end_time = self.time_limit_s + time.perf_counter()
        return time.perf_counter() < self._end_time

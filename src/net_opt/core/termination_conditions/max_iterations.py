import time
from pydantic import BaseModel, PrivateAttr

from net_opt.core.termination_conditions.base_termination_condition import TerminationCondition

class MaxIterations(TerminationCondition, BaseModel):
    iterations: int
    
    _total_iteration_time: float = PrivateAttr(default=0.0)
    _last_start_time: float = PrivateAttr(default_factory=time.perf_counter)

    def check(self, lowest_penalty: float, iteration_n: int) -> bool:
        if not iteration_n:
            self._last_start_time = time.perf_counter()
            return iteration_n < self.iterations
        current_time = time.perf_counter()
        
        # Calculates the duration since the end of previous check to the beginning of current
        iteration_duration = current_time - self._last_start_time
        self._total_iteration_time += iteration_duration
            
        avg_iteration_time = self._total_iteration_time / iteration_n

        print("-" * 40)
        print(f"it: {iteration_n}/{self.iterations}")
        print(f"avg_iter_time: {avg_iteration_time:.4f}s")
            
        remaining_iters = self.iterations - iteration_n
        remaining_time_sec = remaining_iters * avg_iteration_time

        if remaining_time_sec < 0:
            print("projected_rem: --")
        elif remaining_time_sec < 60:
            print(f"projected_rem: {remaining_time_sec:.1f}s")
        elif remaining_time_sec < 3600:
            print(f"projected_rem: {remaining_time_sec / 60:.1f} min")
        else:
            print(f"projected_rem: {remaining_time_sec / 3600:.2f} hr")

        self._last_start_time = current_time

        return iteration_n < self.iterations
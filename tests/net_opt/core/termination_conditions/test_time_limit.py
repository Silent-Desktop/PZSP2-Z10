import pytest
from unittest.mock import patch
from net_opt.core.termination_conditions.time_limit import TimeLimit


class TestTimeLimit:

    @pytest.fixture
    def condition(self):
        return TimeLimit(time_limit_s=10.0)

    def test_initialization(self, condition):
        """
        Verify that iteration 0 correctly sets the internal deadline.
        """
        with patch("time.perf_counter", return_value=100.0):
            result = condition.check(0.0, 0)

            assert result is True
            # Deadline should be Start (100) + Limit (10) = 110
            assert condition._end_time == 110.0

    def test_under_limit(self, condition):
        """
        Verify return is True when time elapsed < limit.
        """
        with patch("time.perf_counter", side_effect=[100.0, 100.0, 105.0]):
            condition.check(0.0, 0)

            # 105 < 110 -> True
            assert condition.check(0.0, 1) is True

    def test_over_limit(self, condition):
        """
        Verify return is False when time elapsed > limit.
        """
        with patch("time.perf_counter", side_effect=[100.0, 100.0, 111.0]):
            condition.check(0.0, 0)

            # 111 > 110 -> False
            assert condition.check(0.0, 5) is False

    def test_boundary_exact(self, condition):
        """
        Edge case: Exact match returns False (< comparison).
        """
        with patch("time.perf_counter", side_effect=[100.0, 100.0, 110.0]):
            condition.check(0.0, 0)

            # 110 == 110 -> False
            assert condition.check(0.0, 2) is False

"""Linear Regression variant of Minimum Expected Time strategy.

Selects the instance with minimum expected queue completion time
using Linear Regression predictions.
"""

from src.algorithms.min_expected_time import MinimumExpectedTimeStrategy


class MinimumExpectedTimeLRStrategy(MinimumExpectedTimeStrategy):
    """Selects the instance with minimum expected queue completion time.

    Uses Linear Regression predictions.
    """

    def get_prediction_type(self) -> str:
        """Return prediction type for Linear Regression."""
        return "linear_regression"

"""Decision Tree variant of Minimum Expected Time strategy.

Selects the instance with minimum expected queue completion time
using Decision Tree predictions.
"""

from src.algorithms.min_expected_time import MinimumExpectedTimeStrategy


class MinimumExpectedTimeDTStrategy(MinimumExpectedTimeStrategy):
    """Selects the instance with minimum expected queue completion time.

    Uses Decision Tree predictions.
    """

    def get_prediction_type(self) -> str:
        """Return prediction type for Decision Tree."""
        return "decision_tree"

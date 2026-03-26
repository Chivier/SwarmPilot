"""Cross-service imports for predictor library classes.

With the unified swarmpilot package, the old src/ namespace collision
is resolved. This module re-exports predictor classes for use within
the scheduler, using direct imports instead of the previous sys.path hack.

Usage:
    from swarmpilot.scheduler.clients._predictor_lib import (
        ExpectErrorPredictor,
        QuantilePredictor,
        ModelStorage,
        ModelCache,
    )
"""

from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.api.core import PredictorLowLevel
from swarmpilot.predictor.models import PlatformInfo
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.preprocessor.adapters import V1PreprocessorAdapter
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_registry import (
    PreprocessorsRegistry,
)
from swarmpilot.predictor.storage.model_storage import ModelStorage
from swarmpilot.predictor.utils.experiment import (
    generate_experiment_prediction,
    is_experiment_mode,
)

# Predictor factory map
PREDICTOR_CLASSES = {
    "expect_error": ExpectErrorPredictor,
    "quantile": QuantilePredictor,
}

__all__ = [
    "PREDICTOR_CLASSES",
    "ExpectErrorPredictor",
    "ModelCache",
    "ModelStorage",
    "PlatformInfo",
    "PredictorLowLevel",
    "PreprocessorChainV2",
    "PreprocessorsRegistry",
    "QuantilePredictor",
    "V1PreprocessorAdapter",
    "generate_experiment_prediction",
    "is_experiment_mode",
]

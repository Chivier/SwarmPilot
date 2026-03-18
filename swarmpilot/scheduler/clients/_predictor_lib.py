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

from swarmpilot.errors import (
    ModelNotFoundError,
    PredictionError,
    PredictorError,
    PredictorValidationError as ValidationError,
    TrainingError,
)
from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.api.core import PredictorLowLevel
from swarmpilot.predictor.models import PlatformInfo
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.preprocessor.adapters import V1PreprocessorAdapter
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
)
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_registry import (
    PreprocessorsRegistry,
)
from swarmpilot.predictor.preprocessor.registry_v2 import (
    PreprocessorsRegistryV2,
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
    "BasePreprocessorV2",
    "ExpectErrorPredictor",
    "FeatureContext",
    "ModelCache",
    "ModelNotFoundError",
    "ModelStorage",
    "PlatformInfo",
    "PredictionError",
    "PredictorError",
    "PredictorLowLevel",
    "PreprocessorChainV2",
    "PreprocessorsRegistry",
    "PreprocessorsRegistryV2",
    "QuantilePredictor",
    "TrainingError",
    "V1PreprocessorAdapter",
    "ValidationError",
    "generate_experiment_prediction",
    "is_experiment_mode",
]

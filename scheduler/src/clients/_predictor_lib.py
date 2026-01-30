"""Bootstrap loader for predictor library classes.

This module resolves the src/ namespace collision between scheduler and
predictor by temporarily swapping sys.path and sys.modules to import
predictor classes, then restoring the scheduler's namespace.

All predictor imports are module-level (no lazy imports), so the entire
dependency tree loads during the swap window.

Usage:
    from src.clients._predictor_lib import (
        ExpectErrorPredictor,
        QuantilePredictor,
        ModelStorage,
        ModelCache,
    )
"""

import sys
from pathlib import Path

# Resolve predictor project root (scheduler/../predictor)
_SCHEDULER_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _SCHEDULER_ROOT.parent
_PREDICTOR_ROOT = _PROJECT_ROOT / "predictor"

if not _PREDICTOR_ROOT.is_dir():
    raise ImportError(
        f"Predictor project not found at {_PREDICTOR_ROOT}. "
        f"Expected sibling directory of scheduler."
    )


def _bootstrap_predictor_imports():
    """Import predictor classes via sys.path manipulation.

    Returns:
        Dict mapping class/function names to their references.
    """
    # 1. Save scheduler's src.* module entries
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            saved_modules[key] = sys.modules.pop(key)

    # 2. Save and modify sys.path
    saved_path = sys.path.copy()
    sys.path.insert(0, str(_PREDICTOR_ROOT))

    try:
        # 3. Import all needed predictor classes (cascade resolves deps)
        from src.api.cache import ModelCache as _ModelCache
        from src.api.core import (
            ModelNotFoundError as _ModelNotFoundError,
            PredictionError as _PredictionError,
            PredictorError as _PredictorError,
            PredictorLowLevel as _PredictorLowLevel,
            TrainingError as _TrainingError,
            ValidationError as _ValidationError,
        )
        from src.models import PlatformInfo as _PlatformInfo
        from src.predictor.expect_error import (
            ExpectErrorPredictor as _ExpectErrorPredictor,
        )
        from src.predictor.quantile import (
            QuantilePredictor as _QuantilePredictor,
        )
        from src.preprocessor.adapters import (
            V1PreprocessorAdapter as _V1PreprocessorAdapter,
        )
        from src.preprocessor.base_preprocessor_v2 import (
            BasePreprocessorV2 as _BasePreprocessorV2,
            FeatureContext as _FeatureContext,
        )
        from src.preprocessor.chain_v2 import (
            PreprocessorChainV2 as _PreprocessorChainV2,
        )
        from src.preprocessor.preprocessors_registry import (
            PreprocessorsRegistry as _PreprocessorsRegistry,
        )
        from src.preprocessor.registry_v2 import (
            PreprocessorsRegistryV2 as _PreprocessorsRegistryV2,
        )
        from src.storage.model_storage import (
            ModelStorage as _ModelStorage,
        )
        from src.utils.experiment import (
            generate_experiment_prediction as _generate_experiment_prediction,
            is_experiment_mode as _is_experiment_mode,
        )

        # 4. Capture references
        imports = {
            "ExpectErrorPredictor": _ExpectErrorPredictor,
            "QuantilePredictor": _QuantilePredictor,
            "ModelStorage": _ModelStorage,
            "ModelCache": _ModelCache,
            "PreprocessorsRegistry": _PreprocessorsRegistry,
            "PlatformInfo": _PlatformInfo,
            "is_experiment_mode": _is_experiment_mode,
            "generate_experiment_prediction": _generate_experiment_prediction,
            # Two-layer API
            "PredictorLowLevel": _PredictorLowLevel,
            "PredictorError": _PredictorError,
            "ModelNotFoundError": _ModelNotFoundError,
            "ValidationError": _ValidationError,
            "TrainingError": _TrainingError,
            "PredictionError": _PredictionError,
            # V2 preprocessor system
            "PreprocessorChainV2": _PreprocessorChainV2,
            "BasePreprocessorV2": _BasePreprocessorV2,
            "FeatureContext": _FeatureContext,
            "PreprocessorsRegistryV2": _PreprocessorsRegistryV2,
            "V1PreprocessorAdapter": _V1PreprocessorAdapter,
        }

        return imports

    finally:
        # 5. Remove predictor's src.* entries from sys.modules
        for key in list(sys.modules.keys()):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]

        # 6. Restore scheduler's original src.* entries and sys.path
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


# Run bootstrap at module load time
_imports = _bootstrap_predictor_imports()

# Export as module-level variables
ExpectErrorPredictor = _imports["ExpectErrorPredictor"]
QuantilePredictor = _imports["QuantilePredictor"]
ModelStorage = _imports["ModelStorage"]
ModelCache = _imports["ModelCache"]
PreprocessorsRegistry = _imports["PreprocessorsRegistry"]
PlatformInfo = _imports["PlatformInfo"]
is_experiment_mode = _imports["is_experiment_mode"]
generate_experiment_prediction = _imports["generate_experiment_prediction"]

# Two-layer API
PredictorLowLevel = _imports["PredictorLowLevel"]
PredictorError = _imports["PredictorError"]
ModelNotFoundError = _imports["ModelNotFoundError"]
ValidationError = _imports["ValidationError"]
TrainingError = _imports["TrainingError"]
PredictionError = _imports["PredictionError"]

# V2 preprocessor system
PreprocessorChainV2 = _imports["PreprocessorChainV2"]
BasePreprocessorV2 = _imports["BasePreprocessorV2"]
FeatureContext = _imports["FeatureContext"]
PreprocessorsRegistryV2 = _imports["PreprocessorsRegistryV2"]
V1PreprocessorAdapter = _imports["V1PreprocessorAdapter"]

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

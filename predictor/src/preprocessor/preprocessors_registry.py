from pathlib import Path
from .semantic_predictor import SemanticPredictor

class PreprocessorsRegistry:
  def __init__(self):
    # Get the predictor directory path (parent of src)
    predictor_dir = Path(__file__).parent.parent.parent
    model_dir = predictor_dir / "preprocessors" / "sematic_35M"

    self.preprocessors = {
      "semantic": SemanticPredictor(
        str(model_dir / "model_35M.pt"),
        str(model_dir / "model_35M.yaml")
      )
    }
    
  def get_preprocessor(self, preprocessor_name: str):
    assert preprocessor_name in self.preprocessors, f"Preprocessor {preprocessor_name} not found"
    return self.preprocessors[preprocessor_name]
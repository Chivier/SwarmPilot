from src.preprocessor.semantic_predictor import SemanticPredictor
from src.preprocessor.preprocessors_registry import PreprocessorsRegistry
import time
from typing import Dict
from pathlib import Path


# Get predictor directory (parent of tests directory)
PREDICTOR_DIR = Path(__file__).parent.parent
MODEL_PATH = str(PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.pt")
CONFIG_PATH = str(PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.yaml")


def test_preprocessor_registry():
  registry = PreprocessorsRegistry()
  assert registry is not None
  assert registry.get_preprocessor('semantic') is not None
  assert isinstance(registry.get_preprocessor('semantic'), SemanticPredictor)
  

def test_semantic_predictor():
  start_time = time.time()
  predictor = SemanticPredictor(model_path=MODEL_PATH, model_config_path=CONFIG_PATH)
  load_end_time = time.time()
  
  assert predictor is not None
  

  predict_start_time = time.time()
  output, remove_origin = predictor.predict(['Hello, world!'])
  predict_end_time = time.time()
  
  assert isinstance(output, dict)
  assert 'output_length' in output
  assert isinstance(output['output_length'], int)
  assert isinstance(remove_origin, bool)
  assert remove_origin is False
  
  print(f"Load time: {load_end_time - start_time} seconds")
  print(f"Predict time: {predict_end_time - predict_start_time} seconds")
  print(f"Predict output length of 'Hello, world!': {output['output_length']}")
  
def test_semantic_predictor_multiple_inputs():
  predictor = SemanticPredictor(model_path=MODEL_PATH, model_config_path=CONFIG_PATH)
  try:
    predictor.predict(['Hello, world!', 'Hello, world!'])
    assert False, "Expected AssertionError for multiple inputs"
  except AssertionError as e:
    assert "SemanticPredictor only supports one input text" in str(e)
    print("Correctly raised AssertionError for multiple inputs")
    
def test_semantic_predictor_with_preprocessor_registry():
  registry = PreprocessorsRegistry()
  predictor = registry.get_preprocessor('semantic')
  assert predictor is not None
  assert isinstance(predictor, SemanticPredictor)
  output, remove_origin = predictor.predict(['Hello, world!'])
  assert isinstance(output, dict)
  assert 'output_length' in output
  assert isinstance(output['output_length'], int)
  assert isinstance(remove_origin, bool)
  assert remove_origin is False
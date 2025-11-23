from typing import List, Dict, Tuple

class BasePreprocessor:
  def __init__(self):
    pass

  def __call__(self, input_text: List[str]) -> Tuple[Dict[str, int], bool]:
    raise NotImplementedError
from typing import Dict, Optional, Tuple

import torch
import yaml
from transformers import AutoTokenizer
from typing import Dict, Optional, Tuple, List

"""
Transformer-based model architectures for output length prediction.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TransformerLengthPredictor(nn.Module):
    """Transformer encoder model for predicting output length."""

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        vocab_size: int = 30522,  # BERT vocab size
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_pretrained: bool = False,
        pretrained_model_name: Optional[str] = None,
    ):
        """
        Initialize the model.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: Size of feed-forward network
            vocab_size: Vocabulary size
            max_position_embeddings: Maximum sequence length
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
            use_pretrained: Whether to use pretrained transformer
            pretrained_model_name: Name of pretrained model if use_pretrained=True
        """
        super().__init__()

        if use_pretrained and pretrained_model_name:
            # Load pretrained transformer
            print(f"Loading pretrained model: {pretrained_model_name}")
            self.transformer = AutoModel.from_pretrained(pretrained_model_name)
            hidden_size = self.transformer.config.hidden_size
        else:
            # Create transformer from scratch
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.num_hidden_layers = num_layers
            config.hidden_size = hidden_size
            config.num_attention_heads = num_attention_heads
            config.intermediate_size = intermediate_size
            config.vocab_size = vocab_size
            config.max_position_embeddings = max_position_embeddings
            config.hidden_dropout_prob = hidden_dropout_prob
            config.attention_probs_dropout_prob = attention_probs_dropout_prob

            self.transformer = AutoModel.from_config(config)
            print(f"Created transformer with {num_layers} layers, "
                  f"hidden_size={hidden_size}")

        # Regression head
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.regressor = nn.Linear(hidden_size, 1)

        # Initialize weights for regression head
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        self.regressor.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Predicted output lengths [batch_size]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout and regression head
        cls_output = self.dropout(cls_output)
        predictions = self.regressor(cls_output).squeeze(-1)  # [batch_size]

        return predictions

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model_from_config(config: Dict) -> TransformerLengthPredictor:
    """
    Create a model from a configuration dictionary.

    Args:
        config: Model configuration

    Returns:
        Initialized model
    """
    model = TransformerLengthPredictor(
        num_layers=config['num_layers'],
        hidden_size=config['hidden_size'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        vocab_size=config.get('vocab_size', 30522),
        max_position_embeddings=config.get('max_position_embeddings', 512),
        hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
        use_pretrained=config.get('use_pretrained', False),
        pretrained_model_name=config.get('pretrained_model_name', None),
    )

    param_count = model.count_parameters()
    print(f"Model created with {param_count:,} parameters "
          f"({param_count / 1e6:.1f}M)")

    return model

def create_model_from_config(config: Dict) -> TransformerLengthPredictor:
    """
    Create a model from a configuration dictionary.

    Args:
        config: Model configuration

    Returns:
        Initialized model
    """
    model = TransformerLengthPredictor(
        num_layers=config['num_layers'],
        hidden_size=config['hidden_size'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        vocab_size=config.get('vocab_size', 30522),
        max_position_embeddings=config.get('max_position_embeddings', 512),
        hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
        use_pretrained=config.get('use_pretrained', False),
        pretrained_model_name=config.get('pretrained_model_name', None),
    )

    param_count = model.count_parameters()
    print(f"Model created with {param_count:,} parameters "
          f"({param_count / 1e6:.1f}M)")

    return model


class SemanticPredictor:
    def __init__(self, model_path: str, model_config_path: str):
        self.model_path = model_path
        with open(model_config_path, 'r') as f:
          self.model_config = yaml.safe_load(f)
        # Load prediction model
        self.load_model()
        
        self.tokenizer_name = self.model_config['data']['tokenizer_name']
        self.max_length = self.model_config['data']['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def load_model(self):
      model_specific_config = self.model_config['model']  
      self.model = create_model_from_config(model_specific_config)
      checkpoint = torch.load(self.model_path, map_location='cpu')
      if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
          self.model.load_state_dict(checkpoint['model_state_dict'])
      else:
          self.model.load_state_dict(checkpoint)
      self.model.eval()
        
    def process_input(self, input_text: str):
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        attention_mask = torch.stack([encoded['attention_mask'].squeeze(0)])
        input_ids = torch.stack([encoded['input_ids'].squeeze(0)])
        return input_ids, attention_mask
    
    def predict(self, input_text: List[str]) -> Tuple[Dict[str, int], bool]:
      assert len(input_text) == 1, "SemanticPredictor only supports one input text"
      input_ids, attention_mask = self.process_input(input_text[0])
      with torch.no_grad():
          predictions = self.model(input_ids, attention_mask)
      prediction = predictions.numpy()[0]
      # Return False: do not remove original input feature
      return {"output_length": int(prediction)}, True
      
    def __call__(self, input_text: str) -> int:
        output, _ = self.predict([input_text])
        return output['output_length']
    
    
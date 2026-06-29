#models/init.py

import torch
from .lstm import LSTMSequenceLabeler

# Registry for future expansion (CNN-LSTM, Transformer)
_MODEL_REGISTRY = {
    "lstm": LSTMSequenceLabeler,
}

def build_model(name, n_freq, n_classes, **kwargs):
    """
    Factory function to instantiate models.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    
    model_class = _MODEL_REGISTRY[name]
    return model_class(n_freq=n_freq, n_classes=n_classes, **kwargs)

def test_model_contract():
    """
    Asserts the input/output contract: (B, T, F) -> (B, T, C)
    """
    B, T, F = 8, 100, 133
    C = 3
    model = build_model("lstm", n_freq=F, n_classes=C)
    
    sample_input = torch.randn(B, T, F)
    output = model(sample_input)
    
    expected_shape = (B, T, C)
    assert output.shape == expected_shape, f"Contract violation! Expected {expected_shape}, got {output.shape}"
    print(f"Model contract verified: {output.shape}")

if __name__ == "__main__":
    test_model_contract()
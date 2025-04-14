import torch.nn as nn

def get_model():
    """User-defined model"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

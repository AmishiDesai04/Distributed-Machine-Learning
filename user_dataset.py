import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(batch_size=4):
    """User-defined dataset"""
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randn(100, 10)  # 100 target values
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

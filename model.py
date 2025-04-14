import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_user_model():
    return ComplexCNN()

def split_model(model, num_workers):
    layers = list(model.children())
    chunk_size = len(layers) // num_workers
    return [nn.Sequential(*layers[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_workers)]

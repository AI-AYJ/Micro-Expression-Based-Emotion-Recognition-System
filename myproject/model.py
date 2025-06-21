import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 112x112
        x = self.pool(F.relu(self.conv2(x)))  # -> 56x56
        x = self.pool(F.relu(self.conv3(x)))  # -> 28x28
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




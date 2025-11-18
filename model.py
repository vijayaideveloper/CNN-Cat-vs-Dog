import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
def __init__(self):
super(ImageClassifier, self).__init__()
self.features = nn.Sequential(
nn.Conv2d(3, 16, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2, stride=2),
nn.Conv2d(16, 32, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2, stride=2),
nn.Conv2d(32, 64, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2, stride=2),
)


self.classifier = nn.Sequential(
nn.Flatten(),
nn.Linear(64 * 28 * 28, 128),
nn.ReLU(),
nn.Linear(128, 2)
)


def forward(self, x):
x = self.features(x)
x = self.classifier(x)
return x
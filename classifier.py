# classifier.py
import torch
import torch.nn as nn
from torchvision import models, transforms


class CelebAClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for a single CelebA binary attribute.
    
    CelebA has 40 binary attributes (smiling, young, blonde, etc.).
    We train one classifier per attribute — kept simple intentionally.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """Returns predicted class index."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def probability(self, x, label):
        """Returns probability of a specific label."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            return probs[:, label]
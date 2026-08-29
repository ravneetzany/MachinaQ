"""Learned CNC-operation classifier — supplements (never replaces) the
rule-based `src.operation_classifier`.

Input is the 8-dim vector `src.operation_classifier_dataset.vectorize()`
produces (primitive type one-hot, log-radius, axis-relationship booleans) —
the same small set of signals the rule engine itself reads, not a richer
point-cloud/graph representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 8
NUM_CLASSES = 6  # len(operation_classifier.Operation.ALL)


class OperationClassifierNet(nn.Module):
    """3-layer MLP: 8 -> 32 -> 16 -> 6 logits over `Operation.ALL`."""

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # (B, num_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)


def save_model(model: OperationClassifierNet, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: OperationClassifierNet, path: str, map_location: str = "cpu") -> OperationClassifierNet:
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model

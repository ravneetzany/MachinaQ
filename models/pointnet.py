"""PointNet models for point cloud classification.

PointNet          — multi-class classifier (holes / bosses / slots / threads / drills)
PointNetBinary    — binary through-hole classifier (through=1, blind=0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PointNetEncoder(nn.Module):
    """Shared feature extractor (1D-conv backbone + global max pool)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)    # (B, 3, N) -> (B, 64, N)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, N)  →  global feature : (B, 1024)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.max(x, 2)[0]           # global max pool → (B, 1024)


class PointNet(nn.Module):
    """Multi-class machining feature classifier.

    Classes (default 5): hole, boss, slot, thread, drill.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.encoder = _PointNetEncoder()
        self.fc1     = nn.Linear(1024, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)                          # (B, 1024)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)                              # (B, num_classes)


class PointNetBinary(nn.Module):
    """Binary through-hole classifier.

    Output logits: shape (B, 2) where index 0=blind, 1=through.
    Use CrossEntropyLoss during training (handles softmax internally).

    The model is trained to detect whether a cylindrical hole feature
    extends all the way through the material (through hole) or terminates
    at a flat bottom inside the part (blind hole).

    Point cloud encoding for training:
        Through hole — cylindrical wall only, BOTH ends open (no flat cap)
        Blind hole   — cylindrical wall + flat disk at one end (the floor)
    """

    def __init__(self):
        super().__init__()
        self.encoder = _PointNetEncoder()
        # Deeper head for the finer geometric distinction
        self.fc1     = nn.Linear(1024, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, 128)
        self.fc4     = nn.Linear(128, 2)        # binary: blind / through
        self.dropout = nn.Dropout(p=0.4)
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.bn3     = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)                          # (B, 1024)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)                              # (B, 2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions: 0=blind, 1=through."""
        return self.forward(x).argmax(dim=1)


class PointNetSeg(nn.Module):
    """Per-point B-Rep face segmentation (Fusion 360 Gallery segmentation dataset).

    Classes (default 8, see nist_sfa/s2.0.0/segment_names.json): ExtrudeSide,
    ExtrudeEnd, CutSide, CutEnd, Fillet, Chamfer, RevolveSide, RevolveEnd.

    Input: (B, in_channels, N) — xyz + unit surface normal per sampled point
    (in_channels=6) by default, since side/end faces of the same primitive
    are often only distinguishable by normal direction.
    Output: (B, num_classes, N) per-point logits.
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Per-point head: local features (64) + broadcast global feature (1024)
        self.seg1 = nn.Conv1d(1024 + 64, 512, 1)
        self.seg2 = nn.Conv1d(512, 256, 1)
        self.seg3 = nn.Conv1d(256, 128, 1)
        self.seg4 = nn.Conv1d(128, num_classes, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(2)
        x = F.relu(self.bn1(self.conv1(x)))
        local = F.relu(self.bn2(self.conv2(x)))         # (B, 64, N)
        x = F.relu(self.bn3(self.conv3(local)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        global_feat = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, n)  # (B, 1024, N)

        x = torch.cat([local, global_feat], dim=1)       # (B, 1088, N)
        x = F.relu(self.bns1(self.seg1(x)))
        x = self.dropout(x)
        x = F.relu(self.bns2(self.seg2(x)))
        x = F.relu(self.bns3(self.seg3(x)))
        return self.seg4(x)                              # (B, num_classes, N)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, map_location='cpu'):
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model
"""Training for a 24-class machining-feature PointNet classifier.

Dataset: nist_sfa/stl/<N>_<name>/*.STL — 24 class folders (0_Oring ...
23_6sides_pocket), each holding STL exports of a single milled feature.
Label = the leading folder index. Point clouds are sampled directly from
each STL's triangle vertices (see stl_ingest.py) — no mesh library needed,
since the STL exports here are ASCII.

This is a distinct model from the 5-class hole/boss/slot/thread/drill
PointNet (models.pointnet.PointNet is reused with num_classes=24, but the
checkpoint and label space are separate — see run_train.py's `mfcad24`
branch).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models.pointnet import PointNet, save_model
from .stl_ingest import parse_ascii_stl_vertices, sample_point_cloud, center_and_scale

logger = logging.getLogger(__name__)

_CLASS_DIR_RE = re.compile(r"^(\d+)_(.+)$")


def discover_classes(stl_root: Path) -> List[Tuple[int, str, Path]]:
    """Return sorted [(label, name, dir_path), ...] from nist_sfa/stl's subfolders."""
    classes = []
    for d in sorted(stl_root.iterdir()):
        if not d.is_dir():
            continue
        m = _CLASS_DIR_RE.match(d.name)
        if not m:
            continue
        classes.append((int(m.group(1)), m.group(2), d))
    classes.sort(key=lambda c: c[0])
    return classes


class MFCAD24Dataset(Dataset):
    """Point clouds sampled from nist_sfa/stl, labeled by class-folder index."""

    def __init__(
        self,
        stl_root: Path,
        n_points: int = 1024,
        max_per_class: Optional[int] = None,
        seed: int = 0,
    ):
        self.n_points = n_points
        self.classes = discover_classes(stl_root)
        if not self.classes:
            raise RuntimeError(f"No class folders found under {stl_root}")
        self.num_classes = max(label for label, _, _ in self.classes) + 1
        self.class_names = {label: name for label, name, _ in self.classes}

        rng = np.random.default_rng(seed)
        self.clouds: List[np.ndarray] = []
        self.labels: List[int] = []

        for label, name, d in self.classes:
            files = sorted(d.glob("*.STL")) + sorted(d.glob("*.stl"))
            if max_per_class is not None:
                files = files[:max_per_class]
            n_ok = 0
            for f in files:
                try:
                    verts = parse_ascii_stl_vertices(f)
                except ValueError as e:
                    logger.warning("Skipping %s: %s", f, e)
                    continue
                cloud = sample_point_cloud(verts, n_points=n_points, rng=rng)
                cloud = center_and_scale(cloud)
                self.clouds.append(cloud)
                self.labels.append(label)
                n_ok += 1
            logger.info("  class %2d %-28s %d samples", label, name, n_ok)

        logger.info(
            "MFCAD24Dataset: %d classes, %d samples total", self.num_classes, len(self.clouds)
        )

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, idx: int):
        cloud = torch.from_numpy(self.clouds[idx]).transpose(0, 1)  # (3, N)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cloud, label


def train_mfcad24(
    stl_root: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_points: int = 1024,
    val_split: float = 0.1,
    max_per_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> PointNet:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MFCAD24Dataset(Path(stl_root), n_points=n_points, max_per_class=max_per_class)

    n_val = 0 if val_split <= 0 else max(1, int(val_split * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if n_val > 0 else None

    model = PointNet(num_classes=dataset.num_classes).to(device)
    logger.info(
        "PointNet(mfcad24)  classes=%d  params=%d  device=%s  train=%d  val=%d",
        dataset.num_classes,
        sum(p.numel() for p in model.parameters()),
        device,
        len(train_ds),
        len(val_ds),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for clouds, labels in train_loader:
            clouds, labels = clouds.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clouds)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()

        train_acc = 100.0 * train_correct / max(train_total, 1)

        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for clouds, labels in val_loader:
                    clouds, labels = clouds.to(device), labels.to(device)
                    logits = model(clouds)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = 100.0 * val_correct / max(val_total, 1)
            logger.info(
                "Ep [%3d/%3d]  loss=%.5f  train_acc=%.2f%%  val_acc=%.2f%%",
                epoch + 1, epochs, train_loss / max(train_total, 1), train_acc, val_acc,
            )
            score = val_acc
        else:
            # No held-out split (val_split<=0): track/save against train_acc instead.
            logger.info(
                "Ep [%3d/%3d]  loss=%.5f  train_acc=%.2f%%  (no validation split)",
                epoch + 1, epochs, train_loss / max(train_total, 1), train_acc,
            )
            score = train_acc

        if score >= best_val_acc:
            best_val_acc = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Training complete. Best val_acc: %.2f%%", best_val_acc)
    return model

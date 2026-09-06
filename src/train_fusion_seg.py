"""Training for per-point B-Rep face segmentation on the Fusion 360 Gallery
segmentation dataset (nist_sfa/s2.0.0).

Dataset layout (Autodesk Fusion 360 Gallery, segmentation subset s2.0.0):
    point_clouds/<name>.xyz   — 2048 points, 6 cols (xyz + unit normal)
    point_clouds/<name>.seg   — 2048 per-point labels, one of 8 classes
    segment_names.json        — the 8 class names, indexed 0..7
    train_test.json           — official {"train": [...], "test": [...]} split

Point clouds are already fixed-size (2048 pts) and pre-normalized by the
dataset itself, so no resampling/normalization step is needed here — unlike
stl_ingest.py's synthetic STL point clouds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.pointnet import PointNetSeg, save_model

logger = logging.getLogger(__name__)


class FusionSegDataset(Dataset):
    """Preloads a Fusion 360 Gallery segmentation split into memory."""

    def __init__(self, root: Path, split: str, max_samples: Optional[int] = None):
        root = Path(root)
        split_map = json.loads((root / "train_test.json").read_text())
        names = split_map[split]
        if max_samples is not None:
            names = names[:max_samples]

        pc_dir = root / "point_clouds"
        points = np.empty((len(names), 2048, 6), dtype=np.float32)
        labels = np.empty((len(names), 2048), dtype=np.int64)

        kept = 0
        for name in names:
            xyz_path = pc_dir / f"{name}.xyz"
            seg_path = pc_dir / f"{name}.seg"
            try:
                pts = np.loadtxt(xyz_path, dtype=np.float32)
                seg = np.loadtxt(seg_path, dtype=np.int64)
            except OSError as e:
                logger.warning("Skipping %s: %s", name, e)
                continue
            if pts.shape != (2048, 6) or seg.shape != (2048,):
                logger.warning("Skipping %s: unexpected shape %s/%s", name, pts.shape, seg.shape)
                continue
            points[kept] = pts
            labels[kept] = seg
            kept += 1

        self.points = points[:kept]
        self.labels = labels[:kept]
        logger.info("FusionSegDataset[%s]: %d/%d samples loaded", split, kept, len(names))

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        pts = torch.from_numpy(self.points[idx]).transpose(0, 1)  # (6, 2048)
        lbl = torch.from_numpy(self.labels[idx])                  # (2048,)
        return pts, lbl


def _per_class_iou(conf: np.ndarray) -> np.ndarray:
    """conf[i, j] = count of points with true label i predicted as j."""
    num_classes = conf.shape[0]
    ious = np.full(num_classes, np.nan)
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    return ious


def train_fusion_seg(
    dataset_root: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> PointNetSeg:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(dataset_root)

    class_names: List[str] = json.loads((root / "segment_names.json").read_text())
    num_classes = len(class_names)

    train_ds = FusionSegDataset(root, "train", max_samples=max_train_samples)
    test_ds = FusionSegDataset(root, "test", max_samples=max_test_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = PointNetSeg(num_classes=num_classes, in_channels=6).to(device)
    logger.info(
        "PointNetSeg  classes=%d %s  params=%d  device=%s  train=%d  test=%d",
        num_classes, class_names,
        sum(p.numel() for p in model.parameters()), device,
        len(train_ds), len(test_ds),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for pts, lbl in train_loader:
            pts, lbl = pts.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(pts)                  # (B, C, N)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lbl.numel()
            train_correct += (logits.argmax(dim=1) == lbl).sum().item()
            train_total += lbl.numel()
        scheduler.step()

        model.eval()
        test_correct = 0
        test_total = 0
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for pts, lbl in test_loader:
                pts, lbl = pts.to(device), lbl.to(device)
                logits = model(pts)
                pred = logits.argmax(dim=1)
                test_correct += (pred == lbl).sum().item()
                test_total += lbl.numel()
                p = pred.flatten().cpu().numpy()
                t = lbl.flatten().cpu().numpy()
                np.add.at(conf, (t, p), 1)

        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)
        miou = np.nanmean(_per_class_iou(conf)) * 100.0
        logger.info(
            "Ep [%3d/%3d]  loss=%.5f  train_acc=%.2f%%  test_acc=%.2f%%  mIoU=%.2f%%",
            epoch + 1, epochs, train_loss / max(train_total, 1), train_acc, test_acc, miou,
        )

        if test_acc >= best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
                logger.info("  * saved checkpoint -> %s", checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final per-class IoU report against the restored best checkpoint
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for pts, lbl in test_loader:
            pts, lbl = pts.to(device), lbl.to(device)
            pred = model(pts).argmax(dim=1)
            p = pred.flatten().cpu().numpy()
            t = lbl.flatten().cpu().numpy()
            np.add.at(conf, (t, p), 1)
    ious = _per_class_iou(conf)
    for name, iou in zip(class_names, ious):
        logger.info("  IoU  %-14s %.2f%%", name, iou * 100.0 if not np.isnan(iou) else float("nan"))

    logger.info("Training complete. Best test_acc: %.2f%%  mIoU: %.2f%%",
                best_acc, np.nanmean(ious) * 100.0)
    return model

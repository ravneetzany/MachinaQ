"""Multi-task training for MachinaQUnified.

Dataset
-------
Through-hole samples (feat_type=hole, is_through=1):
    Sourced from NIST STC/FTC STEP files via StepTextParser.
    Same point-cloud generator as train_through_hole.py (85% wall + 15% noise).

Blind-hole samples (feat_type=hole, is_through=0):
    Synthetically generated — identical size distribution to through holes,
    but with a flat circular disk at one end (60% wall + 25% disk + 15% noise).

Loss
----
  total = feat_weight * CE(feat_logits, feat_labels)
        + hole_weight * CE(hole_logits, through_labels)

The feature head currently only sees 'hole' class (label 0).  When training
data for boss / slot / thread / drill becomes available, add samples with the
corresponding labels; the architecture supports num_feature_classes classes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from .parser import StepTextParser
from models.machinaq_unified import (
    MachinaQUnified,
    FEATURE_CLASSES,
    HOLE_CLASS_IDX,
    merge_pretrained_weights,
    load_unified,
)

logger = logging.getLogger(__name__)

NUM_POINTS = 1024

# ASME standard radii (inches) — used when no STEP files parse successfully
_ASME_RADII_IN = [
    0.060, 0.0865, 0.120, 0.125, 0.1405, 0.21875, 0.2969, 0.4375,
]


# ---------------------------------------------------------------------------
# Point-cloud generators  (mirrored from train_through_hole.py)
# ---------------------------------------------------------------------------

def _cylinder_wall(
    radius: float, depth: float, n: int, jitter: float = 0.05
) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n)
    z     = np.random.uniform(0, depth, n)
    x = radius * np.cos(theta) + np.random.normal(0, jitter * radius, n)
    y = radius * np.sin(theta) + np.random.normal(0, jitter * radius, n)
    return np.stack([x, y, z], axis=1)


def _flat_disk(
    radius: float, z_pos: float, n: int, jitter: float = 0.03
) -> np.ndarray:
    r     = radius * np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta) + np.random.normal(0, jitter * radius, n)
    y = r * np.sin(theta) + np.random.normal(0, jitter * radius, n)
    z = np.full(n, z_pos) + np.random.normal(0, jitter * 0.5, n)
    return np.stack([x, y, z], axis=1)


def _noise(radius: float, depth: float, n: int) -> np.ndarray:
    s = radius * 1.2
    return np.random.uniform([-s, -s, 0], [s, s, depth], (n, 3))


def _normalise(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-8
    return (pts / scale).astype(np.float32)


def _random_depth(radius: float) -> float:
    return radius * 2.0 * np.random.uniform(0.5, 3.0)


def make_through_cloud(radius: float, depth: float) -> np.ndarray:
    """85% wall + 15% noise, small random axis tilt."""
    r_v = radius * (1 + np.random.uniform(-0.15, 0.15))
    d_v = depth  * (1 + np.random.uniform(-0.20, 0.20))
    jit = np.random.uniform(0.02, 0.08)
    n_w = int(NUM_POINTS * 0.85)
    pts = np.vstack([
        _cylinder_wall(r_v, d_v, n_w, jitter=jit),
        _noise(r_v, d_v, NUM_POINTS - n_w),
    ])[:NUM_POINTS]
    a = np.random.uniform(-0.175, 0.175)
    c, s = np.cos(a), np.sin(a)
    pts  = (np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ pts.T).T
    return _normalise(pts)


def make_blind_cloud(radius: float, depth: float) -> np.ndarray:
    """60% wall + 25% flat disk at bottom + 15% noise, small axis tilt."""
    r_v = radius * (1 + np.random.uniform(-0.15, 0.15))
    d_v = depth  * (1 + np.random.uniform(-0.20, 0.20))
    jit = np.random.uniform(0.02, 0.08)
    n_w = int(NUM_POINTS * 0.60)
    n_d = int(NUM_POINTS * 0.25)
    pts = np.vstack([
        _cylinder_wall(r_v, d_v, n_w, jitter=jit),
        _flat_disk(r_v, z_pos=d_v, n=n_d, jitter=jit * 0.5),
        _noise(r_v, d_v, NUM_POINTS - n_w - n_d),
    ])[:NUM_POINTS]
    a = np.random.uniform(-0.175, 0.175)
    c, s = np.cos(a), np.sin(a)
    pts  = (np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ pts.T).T
    return _normalise(pts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedHoleDataset(Dataset):
    """Multi-task dataset for MachinaQUnified.

    Each sample: (point_cloud (3,1024), feat_label, through_label)

    feat_label    : int in range(num_feature_classes)  — currently always 0 (hole)
    through_label : 0 = blind,  1 = through
    """

    def __init__(
        self,
        step_files:       List[str],
        aug_factor:       int   = 20,
        blind_multiplier: float = 1.0,
    ):
        self.clouds:         List[np.ndarray] = []
        self.feat_labels:    List[int]        = []
        self.through_labels: List[int]        = []

        through_radii: List[float] = []

        # ── Real through holes from STEP files ───────────────────────────────
        for path in step_files:
            try:
                parser = StepTextParser()
                parser.parse_file(path)
                n_accepted = 0
                for hole in parser.features.holes:
                    if not hole.get('is_through', False):
                        continue
                    r = hole['radius']
                    through_radii.append(r)
                    for _ in range(aug_factor):
                        d  = _random_depth(r)
                        pc = make_through_cloud(r, d)
                        self.clouds.append(pc)
                        self.feat_labels.append(HOLE_CLASS_IDX)
                        self.through_labels.append(1)
                        n_accepted += 1
                logger.info(
                    f"  {Path(path).name}: {n_accepted} through-hole augmentations"
                )
            except Exception as exc:
                logger.warning(f"  Skipping {Path(path).name}: {exc}")

        # Fallback: seed from ASME standard radii if no real holes were found
        if not through_radii:
            logger.warning(
                "No through holes parsed from STEP files. "
                "Seeding from ASME standard radii."
            )
            repeat = aug_factor // len(_ASME_RADII_IN) + 1
            through_radii = (_ASME_RADII_IN * repeat)[:aug_factor * 5]
            for r in through_radii:
                d  = _random_depth(r)
                pc = make_through_cloud(r, d)
                self.clouds.append(pc)
                self.feat_labels.append(HOLE_CLASS_IDX)
                self.through_labels.append(1)

        # ── Synthetic blind holes (balanced) ─────────────────────────────────
        n_through = self.through_labels.count(1)
        n_blind   = int(n_through * blind_multiplier)
        blind_radii = np.random.choice(through_radii, size=n_blind, replace=True)
        for r in blind_radii:
            d  = _random_depth(float(r))
            pc = make_blind_cloud(float(r), d)
            self.clouds.append(pc)
            self.feat_labels.append(HOLE_CLASS_IDX)
            self.through_labels.append(0)

        logger.info(
            f"UnifiedHoleDataset ready: "
            f"{n_through} through + {n_blind} blind = {len(self.clouds)} total"
        )

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pc        = torch.from_numpy(self.clouds[idx]).T          # (3, N)
        feat_lbl  = torch.tensor(self.feat_labels[idx],    dtype=torch.long)
        hole_lbl  = torch.tensor(self.through_labels[idx], dtype=torch.long)
        return pc, feat_lbl, hole_lbl


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_unified(
    step_files:          List[str],
    epochs:              int   = 30,
    batch_size:          int   = 32,
    lr:                  float = 1e-3,
    aug_factor:          int   = 20,
    feat_weight:         float = 0.5,
    hole_weight:         float = 1.0,
    pretrained_unified:  Optional[str] = None,
    pretrained_pointnet: Optional[str] = None,
    pretrained_binary:   Optional[str] = None,
) -> MachinaQUnified:
    """Train MachinaQUnified with multi-task loss.

    Args:
        step_files:          STEP file paths (STC / FTC — through holes)
        epochs:              max training epochs
        batch_size:          mini-batch size
        lr:                  initial learning rate (cosine-annealed)
        aug_factor:          augmentations per real hole sample
        feat_weight:         loss weight for feature-type head
        hole_weight:         loss weight for through/blind head
        pretrained_unified:  load an existing unified checkpoint to fine-tune
        pretrained_pointnet: PointNet weights to seed encoder + feat head
        pretrained_binary:   PointNetBinary weights to seed encoder + hole head

    Returns:
        Trained MachinaQUnified.
    """
    logger.info("Building UnifiedHoleDataset ...")
    dataset = UnifiedHoleDataset(step_files, aug_factor=aug_factor)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — no valid STEP files.")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MachinaQUnified(num_feature_classes=len(FEATURE_CLASSES)).to(device)

    # ── Weight initialisation ─────────────────────────────────────────────────
    if pretrained_unified and os.path.exists(pretrained_unified):
        ckpt = load_unified(pretrained_unified, device=str(device))
        model.load_state_dict(ckpt.state_dict())
        logger.info(f"Fine-tuning from unified checkpoint: {pretrained_unified}")
    elif pretrained_pointnet or pretrained_binary:
        model = merge_pretrained_weights(
            model, pretrained_pointnet, pretrained_binary, str(device)
        )
        logger.info("Weights merged from pre-trained PointNet / PointNetBinary")

    logger.info(
        f"MachinaQUnified  params={model.param_count():,}  device={device}"
    )

    # ── Class-weighted losses ─────────────────────────────────────────────────
    n_through = dataset.through_labels.count(1)
    n_blind   = dataset.through_labels.count(0)
    w_t = len(dataset) / (2 * max(n_through, 1))
    w_b = len(dataset) / (2 * max(n_blind,   1))
    hole_weights = torch.tensor([w_b, w_t], dtype=torch.float).to(device)

    feat_criterion = nn.CrossEntropyLoss()
    hole_criterion = nn.CrossEntropyLoss(weight=hole_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss    = float('inf')
    patience_ctr = 0
    MAX_PATIENCE = 6

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_feat = correct_hole = total = 0

        for clouds, feat_lbls, hole_lbls in loader:
            clouds    = clouds.to(device)
            feat_lbls = feat_lbls.to(device)
            hole_lbls = hole_lbls.to(device)

            optimizer.zero_grad()
            feat_logits, hole_logits = model(clouds)

            loss = (feat_weight * feat_criterion(feat_logits, feat_lbls) +
                    hole_weight * hole_criterion(hole_logits, hole_lbls))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss   += loss.item()
            correct_feat += (feat_logits.argmax(1) == feat_lbls).sum().item()
            correct_hole += (hole_logits.argmax(1) == hole_lbls).sum().item()
            total        += clouds.size(0)

        scheduler.step()
        avg_loss  = total_loss / len(loader)
        acc_feat  = 100.0 * correct_feat / max(total, 1)
        acc_hole  = 100.0 * correct_hole / max(total, 1)
        cur_lr    = scheduler.get_last_lr()[0]

        logger.info(
            f"Ep [{epoch+1:3d}/{epochs}]  "
            f"loss={avg_loss:.5f}  "
            f"feat_acc={acc_feat:.2f}%  "
            f"hole_acc={acc_hole:.2f}%  "
            f"lr={cur_lr:.6f}"
        )

        if avg_loss < best_loss:
            best_loss    = avg_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= MAX_PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(patience={MAX_PATIENCE})"
                )
                break

    logger.info(f"Training complete. Best loss: {best_loss:.5f}")
    return model

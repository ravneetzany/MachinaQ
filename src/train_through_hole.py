"""Training pipeline for through-hole vs blind-hole binary classification.

Dataset strategy
----------------
Through holes (label=1):
    Sourced from NIST STC STEP files (inch units, sheet-metal parts).
    All cylinders in STC files are by design through holes — the holes
    pierce the full thickness of the sheet.

Blind holes (label=0):
    Synthetically generated.  A blind hole point cloud has:
        - cylindrical wall points (same distribution as through holes)
        - a flat circular disk at the closed end (the hole floor)
    This geometric contrast — presence vs absence of an end cap — is the
    signal the model learns to recognise.

Point-cloud shape (1 024 points, shape (3, 1024)):
    Through:  ~85 % cylindrical wall  +  ~15 % Gaussian noise
    Blind:    ~60 % cylindrical wall  +  ~25 % flat disk  +  ~15 % noise

The radius and depth are drawn from ASME-standard sizes so the model
generalises to real part geometries.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from .parser import StepTextParser
from models.pointnet import PointNetBinary, save_model

logger = logging.getLogger(__name__)

NUM_POINTS = 1024  # points per cloud


# ---------------------------------------------------------------------------
# Point-cloud generators
# ---------------------------------------------------------------------------

def _cylinder_wall(radius: float, depth: float, n: int,
                   jitter: float = 0.05) -> np.ndarray:
    """Random points on a cylinder wall of given radius and depth (z-axis)."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    z     = np.random.uniform(0, depth, n)
    x     = radius * np.cos(theta) + np.random.normal(0, jitter * radius, n)
    y     = radius * np.sin(theta) + np.random.normal(0, jitter * radius, n)
    return np.stack([x, y, z], axis=1)


def _flat_disk(radius: float, z_pos: float, n: int,
               jitter: float = 0.03) -> np.ndarray:
    """Random points on a flat circular disk (hole floor) at height z_pos."""
    r     = radius * np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    x     = r * np.cos(theta) + np.random.normal(0, jitter * radius, n)
    y     = r * np.sin(theta) + np.random.normal(0, jitter * radius, n)
    z     = np.full(n, z_pos) + np.random.normal(0, jitter * 0.5, n)
    return np.stack([x, y, z], axis=1)


def _noise(radius: float, depth: float, n: int) -> np.ndarray:
    """Uniform noise inside the hole volume (scanner clutter)."""
    scale = radius * 1.2
    return np.random.uniform(
        [-scale, -scale, 0],
        [ scale,  scale, depth],
        (n, 3)
    )


def _normalise(pts: np.ndarray) -> np.ndarray:
    """Centre and unit-scale a point cloud."""
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-8
    return (pts / scale).astype(np.float32)


def make_through_cloud(radius: float, depth: float,
                       augment_idx: int = 0) -> np.ndarray:
    """Generate a point cloud for a through hole (no end cap).

    Distribution: 85 % wall, 15 % noise.
    Augmentations include radius variation, depth variation, axis tilt,
    and point jitter.
    """
    # Augmentation
    r_var   = radius  * (1 + np.random.uniform(-0.15, 0.15))
    d_var   = depth   * (1 + np.random.uniform(-0.20, 0.20))
    jitter  = np.random.uniform(0.02, 0.08)

    n_wall  = int(NUM_POINTS * 0.85)
    n_noise = NUM_POINTS - n_wall

    pts = np.vstack([
        _cylinder_wall(r_var, d_var, n_wall,  jitter=jitter),
        _noise(r_var, d_var, n_noise),
    ])[:NUM_POINTS]

    # Random axis tilt (small, ≤10°)
    angle = np.random.uniform(-0.175, 0.175)   # radians
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pts = (rot @ pts.T).T

    return _normalise(pts)


def make_blind_cloud(radius: float, depth: float,
                     augment_idx: int = 0) -> np.ndarray:
    """Generate a point cloud for a blind hole (flat floor at one end).

    Distribution: 60 % wall, 25 % floor disk, 15 % noise.
    """
    r_var   = radius * (1 + np.random.uniform(-0.15, 0.15))
    d_var   = depth  * (1 + np.random.uniform(-0.20, 0.20))
    jitter  = np.random.uniform(0.02, 0.08)

    n_wall  = int(NUM_POINTS * 0.60)
    n_disk  = int(NUM_POINTS * 0.25)
    n_noise = NUM_POINTS - n_wall - n_disk

    pts = np.vstack([
        _cylinder_wall(r_var, d_var, n_wall,  jitter=jitter),
        _flat_disk(r_var, z_pos=d_var, n=n_disk, jitter=jitter * 0.5),
        _noise(r_var, d_var, n_noise),
    ])[:NUM_POINTS]

    # Random axis tilt (small)
    angle = np.random.uniform(-0.175, 0.175)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pts = (rot @ pts.T).T

    return _normalise(pts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ASME standard hole radii (inches) from NIST STC files — used to seed sizes
_ASME_RADII_IN = [0.060, 0.0865, 0.120, 0.125, 0.1405, 0.21875, 0.2969, 0.4375]

# Corresponding typical depth range (inches): 0.5–3× diameter
def _random_depth(radius: float) -> float:
    return radius * 2.0 * np.random.uniform(0.5, 3.0)


class ThroughHoleDataset(Dataset):
    """Labelled point-cloud dataset: through (1) vs blind (0) holes.

    Through-hole samples are extracted from NIST STC STEP files.
    Blind-hole samples are synthetically generated with matching radii
    so the model cannot cheat on size alone.

    Args:
        step_files       : list of STC STEP file paths (through hole source)
        aug_factor       : augmentations per real through-hole sample
        blind_multiplier : blind samples generated per through sample
                           (1.0 = equal balance; >1 upsamples blind)
    """

    def __init__(
        self,
        step_files: List[str],
        aug_factor: int = 20,
        blind_multiplier: float = 1.0,
    ):
        self.clouds: List[np.ndarray] = []
        self.labels: List[int]        = []

        through_radii: List[float] = []

        # ── Source through holes from STEP files ────────────────────────
        for path in step_files:
            try:
                parser = StepTextParser()
                parser.parse_file(path)
                for hole in parser.features.holes:
                    if not hole.get('is_through', False):
                        continue
                    r = hole['radius']     # snapped ASME/ISO radius
                    # Estimate depth from file metadata not available →
                    # use random depth in realistic range
                    through_radii.append(r)
                    for aug_i in range(aug_factor):
                        d = _random_depth(r)
                        pc = make_through_cloud(r, d, aug_i)
                        self.clouds.append(pc)
                        self.labels.append(1)   # through
                logger.info(
                    f"Through holes from {Path(path).name}: "
                    f"{len(parser.features.holes)} holes accepted"
                )
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")

        # If no real through holes found, seed from ASME standard sizes
        if not through_radii:
            logger.warning(
                "No through holes parsed from STEP files. "
                "Seeding from ASME standard radii."
            )
            through_radii = _ASME_RADII_IN * (aug_factor // len(_ASME_RADII_IN) + 1)
            for r in through_radii[:aug_factor * 5]:
                d  = _random_depth(r)
                pc = make_through_cloud(r, d)
                self.clouds.append(pc)
                self.labels.append(1)

        # ── Synthetic blind holes (balanced with through) ────────────────
        n_through = self.labels.count(1)
        n_blind   = int(n_through * blind_multiplier)
        # Draw radii from the same distribution as through holes to prevent
        # the model from learning size rather than topology
        blind_radii = np.random.choice(through_radii, size=n_blind, replace=True)
        for r in blind_radii:
            d  = _random_depth(float(r))
            pc = make_blind_cloud(float(r), d)
            self.clouds.append(pc)
            self.labels.append(0)   # blind

        logger.info(
            f"Dataset: {n_through} through + {n_blind} blind = "
            f"{len(self.clouds)} total"
        )

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pc    = torch.from_numpy(self.clouds[idx]).T   # (3, N)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pc, label


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_through_hole_classifier(
    step_files: List[str],
    epochs:     int   = 30,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    aug_factor: int   = 20,
) -> PointNetBinary:
    """Train and return a PointNetBinary through-hole classifier.

    Args:
        step_files : STC STEP file paths (through hole source)
        epochs     : training epochs (early-stopped by patience)
        batch_size : mini-batch size
        lr         : initial learning rate (cosine-annealed)
        aug_factor : augmentations per real hole sample
    """
    logger.info("Building through-hole dataset …")
    dataset = ThroughHoleDataset(step_files, aug_factor=aug_factor)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — no valid STEP files.")

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PointNetBinary().to(device)
    logger.info(
        f"PointNetBinary  params={sum(p.numel() for p in model.parameters()):,}"
        f"  device={device}"
    )

    # Class weights: compensate if imbalanced
    n_through = dataset.labels.count(1)
    n_blind   = dataset.labels.count(0)
    w_through = len(dataset) / (2 * max(n_through, 1))
    w_blind   = len(dataset) / (2 * max(n_blind,   1))
    weights   = torch.tensor([w_blind, w_through], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss    = float('inf')
    patience_ctr = 0
    MAX_PATIENCE = 6

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0

        for clouds, labels in loader:
            clouds, labels = clouds.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clouds)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc      = 100 * correct / total
        logger.info(
            f"Ep [{epoch+1:3d}/{epochs}]  "
            f"loss={avg_loss:.5f}  acc={acc:.2f}%  lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if avg_loss < best_loss:
            best_loss    = avg_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= MAX_PATIENCE:
                logger.info(f"Early stop at epoch {epoch+1} (patience={MAX_PATIENCE})")
                break

    logger.info(f"Training complete. Best loss: {best_loss:.5f}")
    return model


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
    )

    nist_dir  = Path(__file__).parent.parent / 'nist_sfa'
    # STC files are sheet metal = all through holes
    stc_files = sorted(nist_dir.glob('nist_stc_*.stp'))
    # Also include FTC (Flat Test Cases) which also have through holes
    ftc_files = sorted(nist_dir.glob('nist_ftc_*.stp'))
    all_files = [str(f) for f in stc_files + ftc_files]

    logger.info(f"STEP files: {len(all_files)} "
                f"(STC={len(stc_files)}, FTC={len(ftc_files)})")

    model = train_through_hole_classifier(
        all_files,
        epochs=30,
        batch_size=32,
        aug_factor=20,
    )

    save_path = Path(__file__).parent.parent / 'outputs' / 'machinaq_through_hole.pth'
    save_model(model, str(save_path))
    logger.info(f"Model saved → {save_path}")

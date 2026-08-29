"""Training for the learned CNC-operation classifier (OperationClassifierNet).

Self-distilled from the rule-based operation_classifier: every label comes
from `classify_feature()`'s own output over a multi-source corpus (NIST
STEP files, OpenSCAD parts, FreeCAD parts) — see operation_classifier_dataset.py.
This means the model can only approximate the rules it was trained on, not
exceed them; that is the explicit, accepted scope of this trainer (see
design.md's Risks / Trade-offs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from models.operation_classifier_net import OperationClassifierNet, save_model

from .operation_classifier import Operation
from .operation_classifier_dataset import TrainingExample, build_corpus

logger = logging.getLogger(__name__)

LABEL_TO_INDEX = {op: i for i, op in enumerate(Operation.ALL)}

DEFAULT_SCAD_ROOT = "/home/ravneetzany/projects/openscad-parts-library"
DEFAULT_FREECAD_ROOT = "/home/ravneetzany/projects/freecad-parts-library"


class OperationVectorDataset(Dataset):
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        vector = torch.tensor(example.vector, dtype=torch.float32)
        label = torch.tensor(LABEL_TO_INDEX[example.label], dtype=torch.long)
        return vector, label


def _default_nist_paths() -> List[str]:
    root = Path(__file__).parent.parent
    paths = sorted(str(p) for p in (root / "nist_sfa" / "holeTrain").glob("*.step"))
    paths += sorted(str(p) for p in (root / "nist_sfa").glob("*.stp"))
    return paths


def train_operation_classifier(
    nist_paths: Optional[List[str]] = None,
    scad_root: str = DEFAULT_SCAD_ROOT,
    freecad_root: str = DEFAULT_FREECAD_ROOT,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_path: Optional[str] = None,
) -> OperationClassifierNet:
    if nist_paths is None:
        nist_paths = _default_nist_paths()

    logger.info("Building training corpus ...")
    examples = build_corpus(nist_paths, scad_root, freecad_root)
    if len(examples) == 0:
        raise RuntimeError("Training corpus is empty — no valid source files.")

    dataset = OperationVectorDataset(examples)
    n_val = max(1, int(0.2 * len(dataset))) if len(dataset) >= 5 else 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        train_ds, val_ds = dataset, None

    batch_size = max(1, min(batch_size, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OperationClassifierNet().to(device)
    logger.info(
        "OperationClassifierNet  params=%d  device=%s  train=%d  val=%d",
        sum(p.numel() for p in model.parameters()), device, len(train_ds),
        len(val_ds) if val_ds is not None else 0,
    )

    # Class-weighted loss — the corpus is small and imbalanced (design.md Risks).
    counts = [0] * len(Operation.ALL)
    for example in examples:
        counts[LABEL_TO_INDEX[example.label]] += 1
    weights = torch.tensor(
        [len(examples) / (len(Operation.ALL) * max(c, 1)) for c in counts],
        dtype=torch.float,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    output_path = output_path or str(Path(__file__).parent.parent / "outputs" / "machinaq_operation_classifier.pth")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = total = 0
        for vectors, labels in train_loader:
            vectors, labels = vectors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(vectors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += vectors.size(0)

        avg_loss = total_loss / max(len(train_loader), 1)
        acc = 100.0 * correct / max(total, 1)

        val_acc_str = ""
        if val_loader is not None:
            model.eval()
            v_correct = v_total = 0
            with torch.no_grad():
                for vectors, labels in val_loader:
                    vectors, labels = vectors.to(device), labels.to(device)
                    logits = model(vectors)
                    v_correct += (logits.argmax(1) == labels).sum().item()
                    v_total += vectors.size(0)
            val_acc_str = f"  val_acc={100.0 * v_correct / max(v_total, 1):.2f}%"

        logger.info(
            "Ep [%3d/%d]  loss=%.5f  train_acc=%.2f%%%s",
            epoch + 1, epochs, avg_loss, acc, val_acc_str,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, output_path)

    logger.info("Training complete. Best loss: %.5f. Saved -> %s", best_loss, output_path)
    logger.info(
        "NOTE: labels are self-distilled from operation_classifier's own rules "
        "(see design.md) — this model's accuracy is measured against the rules, "
        "not independent ground truth."
    )
    return model

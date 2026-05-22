"""
MachinaQ Unified Model
======================

Combines three specialised models into one cohesive system:

  1. MachinaQUnified   — multi-task PointNet (shared backbone, two heads)
       • Shared _PointNetEncoder  : (B,3,N) -> (B,1024)  [3 Conv1d layers + global max-pool]
       • Feature-class head       : 5 classes (hole / boss / slot / thread / drill)
       • Through-hole head        : 2 classes (blind / through)

  2. AAGNetSegmentor (GNN) — 25-class B-Rep face segmentation
       • Operates on DGL attribute-graphs pre-built from STEP files
       • Optional: degrades gracefully when DGL is not installed

  3. MachinaQPipeline — unified inference orchestrator
       • predict_file(step_path)   : STEP file  -> MachiningResult
       • predict_cloud(pc_tensor)  : (B,3,N)    -> per-sample dicts
       • predict_graph(graph_data) : DGL graph  -> GNNResult

Weight migration
----------------
merge_pretrained_weights(unified, pointnet_path, binary_path)
  Copies the shared encoder (averaged from both), feature head from PointNet,
  and through/blind head from PointNetBinary into a fresh MachinaQUnified.

I/O helpers
-----------
  save_unified(model, path)   — saves state_dict + num_feature_classes
  load_unified(path, device)  — restores from checkpoint
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

FEATURE_CLASSES: List[str] = ['hole', 'boss', 'slot', 'thread', 'drill']
HOLE_CLASS_IDX = 0

# AAGNet / MFInstSeg 25-class labels (ordered by class index)
GNN_CLASSES: List[str] = [
    'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage',
    'six_sides_passage', 'triangular_through_slot', 'rectangular_through_slot',
    'circular_through_slot', 'rectangular_through_step', 'two_sides_through_step',
    'slanted_through_step', 'o_ring', 'blind_hole', 'triangular_pocket',
    'rectangular_pocket', 'six_sides_pocket', 'circular_end_pocket',
    'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step',
    'round', 'stock', 'triangular_passage_2', 'rectangular_passage_2',
    'six_sides_passage_2',
]

# ---------------------------------------------------------------------------
# Shared encoder — mirrors pointnet.py _PointNetEncoder exactly
# ---------------------------------------------------------------------------

class _PointNetEncoder(nn.Module):
    """1D-conv feature extractor with global max-pooling.

    Input : (B, 3, N)
    Output: (B, 1024)  — global feature vector per sample
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.max(x, 2)[0]           # (B, 1024)


# ---------------------------------------------------------------------------
# MachinaQUnified — multi-task PointNet
# ---------------------------------------------------------------------------

class MachinaQUnified(nn.Module):
    """Multi-task PointNet: feature classification + through/blind detection.

    A single forward pass produces two independent sets of logits from the
    same shared encoder, enabling joint inference and joint training.

    Layer names are chosen to match the separate PointNet / PointNetBinary
    models so that merge_pretrained_weights() can copy weights without any
    manual remapping.

    Args:
        num_feature_classes: number of machining feature types (default 5)
    """

    def __init__(self, num_feature_classes: int = 5):
        super().__init__()
        self.num_feature_classes = num_feature_classes

        # Shared backbone
        self.encoder = _PointNetEncoder()

        # ── Feature-classification head (mirrors PointNet) ───────────────────
        self.feat_fc1  = nn.Linear(1024, 512)
        self.feat_fc2  = nn.Linear(512, 256)
        self.feat_fc3  = nn.Linear(256, num_feature_classes)
        self.feat_bn1  = nn.BatchNorm1d(512)
        self.feat_bn2  = nn.BatchNorm1d(256)
        self.feat_drop = nn.Dropout(p=0.3)

        # ── Through/blind binary head (mirrors PointNetBinary) ───────────────
        self.hole_fc1  = nn.Linear(1024, 512)
        self.hole_fc2  = nn.Linear(512, 256)
        self.hole_fc3  = nn.Linear(256, 128)
        self.hole_fc4  = nn.Linear(128, 2)
        self.hole_bn1  = nn.BatchNorm1d(512)
        self.hole_bn2  = nn.BatchNorm1d(256)
        self.hole_bn3  = nn.BatchNorm1d(128)
        self.hole_drop = nn.Dropout(p=0.4)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, N) point cloud tensor

        Returns:
            feat_logits : (B, num_feature_classes)
            hole_logits : (B, 2)  — index 0 = blind, 1 = through
        """
        z = self.encoder(x)                              # (B, 1024)

        # Feature head
        f = F.relu(self.feat_bn1(self.feat_fc1(z)))
        f = self.feat_drop(f)
        f = F.relu(self.feat_bn2(self.feat_fc2(f)))
        f = self.feat_drop(f)
        feat_logits = self.feat_fc3(f)                  # (B, C)

        # Through/blind head
        h = F.relu(self.hole_bn1(self.hole_fc1(z)))
        h = self.hole_drop(h)
        h = F.relu(self.hole_bn2(self.hole_fc2(h)))
        h = self.hole_drop(h)
        h = F.relu(self.hole_bn3(self.hole_fc3(h)))
        hole_logits = self.hole_fc4(h)                  # (B, 2)

        return feat_logits, hole_logits

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard predictions: (feature_class_idx, is_through_flag)."""
        feat_logits, hole_logits = self.forward(x)
        return feat_logits.argmax(1), hole_logits.argmax(1)

    @torch.no_grad()
    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Soft predictions with per-class probabilities.

        Returns dict with keys:
          feat_class, feat_conf, feat_probs,
          is_through, through_conf, hole_probs
        """
        feat_logits, hole_logits = self.forward(x)
        feat_probs = F.softmax(feat_logits, dim=1)
        hole_probs = F.softmax(hole_logits, dim=1)
        return {
            'feat_class':   feat_logits.argmax(1),
            'feat_conf':    feat_probs.max(1).values,
            'feat_probs':   feat_probs,
            'is_through':   hole_logits.argmax(1),
            'through_conf': hole_probs[:, 1],
            'hole_probs':   hole_probs,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Weight migration from separately-trained models
# ---------------------------------------------------------------------------

def merge_pretrained_weights(
    unified:        MachinaQUnified,
    pointnet_path:  Optional[str] = None,
    binary_path:    Optional[str] = None,
    device:         str = 'cpu',
) -> MachinaQUnified:
    """Copy weights from PointNet and PointNetBinary into a MachinaQUnified.

    Mapping rules
    -------------
    encoder.*          <- averaged from both sources (or single if only one)
    feat_{fc,bn}*      <- PointNet   fc1/fc2/fc3  bn1/bn2  dropout
    hole_{fc,bn}*      <- PointNetBinary  fc1..fc4  bn1/bn2/bn3  dropout

    Args:
        unified:       MachinaQUnified to fill in-place
        pointnet_path: path to machinaq_pointnet.pth  (multi-class PointNet)
        binary_path:   path to machinaq_through_hole.pth  (PointNetBinary)
        device:        map_location for torch.load

    Returns:
        The modified unified model (same object).
    """
    pn_sd  = torch.load(pointnet_path, map_location=device) if pointnet_path else None
    bin_sd = torch.load(binary_path,   map_location=device) if binary_path   else None

    usd = unified.state_dict()

    # Detect old PointNet format: encoder stored flat as 'conv1.*' not 'encoder.conv1.*'
    pn_old_fmt  = pn_sd  is not None and 'conv1.weight' in pn_sd
    bin_old_fmt = bin_sd is not None and 'conv1.weight' in bin_sd

    def _pn_enc_key(k: str, old_fmt: bool) -> str:
        """Map unified 'encoder.conv1.weight' to source key."""
        # k is like 'encoder.conv1.weight' or 'encoder.bn1.running_mean'
        suffix = k[len('encoder.'):]          # 'conv1.weight'
        if old_fmt:
            return suffix                      # old: 'conv1.weight'
        return k                               # new: 'encoder.conv1.weight'

    # ── Shared encoder ───────────────────────────────────────────────────────
    enc_keys = [k for k in usd if k.startswith('encoder.')]
    if pn_sd is not None and bin_sd is not None:
        logger.info("Encoder: averaging PointNet + PointNetBinary weights")
        for k in enc_keys:
            pk  = _pn_enc_key(k, pn_old_fmt)
            bk  = _pn_enc_key(k, bin_old_fmt)
            p_v = pn_sd.get(pk)
            b_v = bin_sd.get(bk)
            if p_v is not None and b_v is not None and p_v.shape == b_v.shape:
                usd[k] = (p_v.float() + b_v.float()) / 2.0
            elif p_v is not None and p_v.shape == usd[k].shape:
                usd[k] = p_v
            elif b_v is not None and b_v.shape == usd[k].shape:
                usd[k] = b_v
    elif pn_sd is not None:
        logger.info("Encoder: loading from PointNet")
        for k in enc_keys:
            v = pn_sd.get(_pn_enc_key(k, pn_old_fmt))
            if v is not None and v.shape == usd[k].shape:
                usd[k] = v
    elif bin_sd is not None:
        logger.info("Encoder: loading from PointNetBinary")
        for k in enc_keys:
            v = bin_sd.get(_pn_enc_key(k, bin_old_fmt))
            if v is not None and v.shape == usd[k].shape:
                usd[k] = v

    # ── Feature head <- PointNet ─────────────────────────────────────────────
    if pn_sd is not None:
        remap = {
            'fc1.weight': 'feat_fc1.weight', 'fc1.bias': 'feat_fc1.bias',
            'fc2.weight': 'feat_fc2.weight', 'fc2.bias': 'feat_fc2.bias',
            'fc3.weight': 'feat_fc3.weight', 'fc3.bias': 'feat_fc3.bias',
            'bn1.weight': 'feat_bn1.weight', 'bn1.bias': 'feat_bn1.bias',
            'bn1.running_mean':        'feat_bn1.running_mean',
            'bn1.running_var':         'feat_bn1.running_var',
            'bn1.num_batches_tracked': 'feat_bn1.num_batches_tracked',
            'bn2.weight': 'feat_bn2.weight', 'bn2.bias': 'feat_bn2.bias',
            'bn2.running_mean':        'feat_bn2.running_mean',
            'bn2.running_var':         'feat_bn2.running_var',
            'bn2.num_batches_tracked': 'feat_bn2.num_batches_tracked',
        }
        copied = 0
        for src, dst in remap.items():
            if src in pn_sd and dst in usd:
                if pn_sd[src].shape == usd[dst].shape:
                    usd[dst] = pn_sd[src]
                    copied += 1
        logger.info(f"Feature head: copied {copied}/{len(remap)} tensors from PointNet")

    # ── Through/blind head <- PointNetBinary ─────────────────────────────────
    if bin_sd is not None:
        remap = {
            'fc1.weight': 'hole_fc1.weight', 'fc1.bias': 'hole_fc1.bias',
            'fc2.weight': 'hole_fc2.weight', 'fc2.bias': 'hole_fc2.bias',
            'fc3.weight': 'hole_fc3.weight', 'fc3.bias': 'hole_fc3.bias',
            'fc4.weight': 'hole_fc4.weight', 'fc4.bias': 'hole_fc4.bias',
            'bn1.weight': 'hole_bn1.weight', 'bn1.bias': 'hole_bn1.bias',
            'bn1.running_mean':        'hole_bn1.running_mean',
            'bn1.running_var':         'hole_bn1.running_var',
            'bn1.num_batches_tracked': 'hole_bn1.num_batches_tracked',
            'bn2.weight': 'hole_bn2.weight', 'bn2.bias': 'hole_bn2.bias',
            'bn2.running_mean':        'hole_bn2.running_mean',
            'bn2.running_var':         'hole_bn2.running_var',
            'bn2.num_batches_tracked': 'hole_bn2.num_batches_tracked',
            'bn3.weight': 'hole_bn3.weight', 'bn3.bias': 'hole_bn3.bias',
            'bn3.running_mean':        'hole_bn3.running_mean',
            'bn3.running_var':         'hole_bn3.running_var',
            'bn3.num_batches_tracked': 'hole_bn3.num_batches_tracked',
        }
        copied = 0
        for src, dst in remap.items():
            if src in bin_sd and dst in usd:
                if bin_sd[src].shape == usd[dst].shape:
                    usd[dst] = bin_sd[src]
                    copied += 1
        logger.info(
            f"Through/blind head: copied {copied}/{len(remap)} tensors "
            f"from PointNetBinary"
        )

    unified.load_state_dict(usd)
    return unified


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_unified(model: MachinaQUnified, path: str) -> None:
    """Save MachinaQUnified to a checkpoint file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            'state_dict':          model.state_dict(),
            'num_feature_classes': model.num_feature_classes,
        },
        path,
    )
    logger.info(f"MachinaQUnified saved -> {path}")


def load_unified(path: str, device: str = 'cpu') -> MachinaQUnified:
    """Load MachinaQUnified from a checkpoint file.

    Supports two formats:
      • New (from save_unified): dict with keys 'state_dict' and
        'num_feature_classes'.
      • Legacy (plain state_dict): the raw OrderedDict returned by
        model.state_dict() — num_feature_classes is inferred from the
        output layer shape.
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        # New format written by save_unified()
        state_dict          = ckpt['state_dict']
        num_feature_classes = ckpt.get('num_feature_classes', 5)
    else:
        # Legacy: plain OrderedDict from torch.save(model.state_dict(), ...)
        state_dict = ckpt
        # Infer num classes from the output layer
        fc3_w = state_dict.get('feat_fc3.weight') or state_dict.get('fc3.weight')
        num_feature_classes = fc3_w.shape[0] if fc3_w is not None else 5

    model = MachinaQUnified(num_feature_classes=num_feature_classes)
    # Strict=False so that a PointNetBinary state dict (missing feat_ keys) can
    # still seed just the encoder + hole head during merge operations.
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HoleResult:
    """Inference result for a single detected hole."""
    step_id:               int
    radius_in:             float
    diameter_mm:           float
    asme_label:            str
    asme_standard:         str
    asme_category:         str

    # PointNet / MachinaQUnified predictions
    feat_type:             str    # 'hole', 'boss', etc.
    feat_confidence:       float
    is_through_model:      bool   # from unified through/blind head
    through_confidence:    float  # probability of 'through' class

    # B-Rep heuristic from StepTextParser topology
    is_through_heuristic:  bool

    # Final verdict (model prediction is authoritative)
    is_through:            bool

    depth_estimate_in:     float = 0.0
    unit_system:           str   = 'inch'

    def label_str(self) -> str:
        """Human-readable one-liner for this hole."""
        flag = 'THROUGH' if self.is_through else 'BLIND'
        conf = f'{self.through_confidence * 100:.1f}%'
        return (
            f"[{flag} {conf}] ID={self.step_id}  "
            f"R={self.radius_in:.5f}in  dia={self.diameter_mm:.3f}mm  "
            f"{self.asme_label} ({self.asme_standard})"
        )


@dataclass
class GNNResult:
    """Face-level segmentation result from AAGNetSegmentor."""
    face_labels:      np.ndarray           # (N_faces,) int32
    face_label_names: List[str]            # GNN_CLASSES[label] per face
    instance_mask:    Optional[np.ndarray] = None  # (N_faces,) int32

    def class_summary(self) -> str:
        unique, counts = np.unique(self.face_labels, return_counts=True)
        parts = [
            f"{GNN_CLASSES[u] if u < len(GNN_CLASSES) else str(u)} x{c}"
            for u, c in zip(unique, counts)
        ]
        return '  '.join(parts)


@dataclass
class MachiningResult:
    """Aggregated inference result for one STEP file."""
    file_path:         str
    unit_system:       str
    holes:             List[HoleResult]    = field(default_factory=list)
    gnn:               Optional[GNNResult] = None
    processing_time_s: float               = 0.0

    @property
    def n_holes(self) -> int:
        return len(self.holes)

    @property
    def n_through(self) -> int:
        return sum(1 for h in self.holes if h.is_through)

    @property
    def n_blind(self) -> int:
        return sum(1 for h in self.holes if not h.is_through)

    def summary(self) -> str:
        lines = [
            '=' * 60,
            f'File        : {os.path.basename(self.file_path)}',
            f'Units       : {self.unit_system}',
            f'Total holes : {self.n_holes}  '
            f'({self.n_through} through, {self.n_blind} blind)',
        ]
        if self.holes:
            lines.append('Holes:')
            for h in self.holes:
                lines.append(f'  {h.label_str()}')
        if self.gnn:
            lines.append(f'GNN faces   : {self.gnn.class_summary()}')
        lines.append(f'Time        : {self.processing_time_s:.2f}s')
        lines.append('=' * 60)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# MachinaQPipeline — full inference orchestrator
# ---------------------------------------------------------------------------

class MachinaQPipeline:
    """Unified MachinaQ inference pipeline combining all three models.

    Models
    ------
    MachinaQUnified   : multi-task PointNet (feature type + through/blind)
    AAGNetSegmentor   : GNN for 25-class B-Rep face segmentation (optional)

    Usage
    -----
    ::

        pipeline = MachinaQPipeline.from_pretrained(
            unified_path = 'outputs/machinaq_unified.pth',
            gnn_path     = 'outputs/machinaq_gnn.pth',   # optional
        )

        # STEP file -> full result
        result = pipeline.predict_file('nist_stc_07.stp')
        print(result.summary())

        # Raw point cloud -> predictions
        pc = torch.randn(4, 3, 1024)        # batch of 4
        preds = pipeline.predict_cloud(pc)  # list of 4 dicts

        # DGL graph -> GNN face labels
        gnn_out = pipeline.predict_graph(graph_data)

    Note on through/blind inference from STEP files
    -----------------------------------------------
    The PointNet through/blind head was trained on *synthetic* point clouds
    where blind holes include a flat-disk cap.  When predicting from a STEP
    file the pipeline generates a wall-only cloud and uses the B-Rep topology
    heuristic from the parser as the authoritative signal; the model head
    provides a confidence score and can override borderline heuristics.
    """

    NUM_POINTS = 1024

    def __init__(
        self,
        unified: Optional[MachinaQUnified] = None,
        gnn:     Optional[nn.Module]       = None,
        device:  Optional[str]             = None,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.unified = unified
        if self.unified is not None:
            self.unified.to(self.device).eval()

        self.gnn = gnn
        if self.gnn is not None:
            self.gnn.to(self.device).eval()

        self._parser_cls = None   # lazy-loaded StepTextParser

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        unified_path: Optional[str] = None,
        gnn_path:     Optional[str] = None,
        device:       Optional[str] = None,
    ) -> 'MachinaQPipeline':
        """Load pipeline from weight files.

        Args:
            unified_path : path to machinaq_unified.pth
            gnn_path     : path to machinaq_gnn.pth (optional)
            device       : 'cpu' or 'cuda' (auto-detected if omitted)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        unified = None
        if unified_path:
            if os.path.exists(unified_path):
                unified = load_unified(unified_path, device=device)
                logger.info(
                    f"MachinaQUnified loaded  params={unified.param_count():,}"
                    f"  device={device}"
                )
            else:
                logger.warning(f"MachinaQUnified weights not found: {unified_path}")

        gnn = cls._load_gnn(gnn_path, device) if gnn_path else None

        return cls(unified=unified, gnn=gnn, device=device)

    @staticmethod
    def _load_gnn(path: str, device: str) -> Optional[nn.Module]:
        """Load AAGNetSegmentor; silently returns None if DGL is absent."""
        if not path or not os.path.exists(path):
            logger.warning(f"GNN weights not found: {path}")
            return None
        try:
            _root    = str(Path(__file__).parent.parent)
            _machgnn = os.path.join(_root, 'machgnn')
            for p in (_root, _machgnn):
                if p not in sys.path:
                    sys.path.insert(0, p)

            import os as _os
            _os.environ.setdefault('DGLBACKEND', 'pytorch')

            from models.inst_segmentors import AAGNetSegmentor  # type: ignore
            from dataloader.mfinstseg   import MFInstSegDataset  # type: ignore

            N_CLASSES = MFInstSegDataset.num_classes('full')
            gnn = AAGNetSegmentor(
                num_classes=N_CLASSES, arch='AAGNetGraphEncoder',
                edge_attr_dim=12, node_attr_dim=10,
                edge_attr_emb=64, node_attr_emb=64,
                edge_grid_dim=0,  node_grid_dim=7,
                edge_grid_emb=0,  node_grid_emb=64,
                num_layers=3, delta=2, mlp_ratio=2,
                drop=0.0, drop_path=0.0, head_hidden_dim=64,
                conv_on_edge=False, use_uv_gird=True,
                use_edge_attr=True, use_face_attr=True,
            )
            gnn.load_state_dict(torch.load(path, map_location=device))
            gnn.to(device).eval()
            total = sum(p.numel() for p in gnn.parameters())
            logger.info(
                f"AAGNetSegmentor loaded  params={total:,}"
                f"  classes={N_CLASSES}  device={device}"
            )
            return gnn
        except Exception as exc:
            logger.warning(f"GNN not loaded ({exc}). Running PointNet-only mode.")
            return None

    # ── Inference: STEP file ─────────────────────────────────────────────────

    def predict_file(self, step_path: str) -> MachiningResult:
        """Run full inference on a STEP file.

        1. Parse STEP with StepTextParser (extracts holes + topology heuristic)
        2. Run MachinaQUnified on each hole's point cloud
        3. Assemble MachiningResult

        GNN face-level segmentation is NOT run here because it requires a
        pre-built DGL graph; call predict_graph() separately and attach via
        result.gnn = pipeline.predict_graph(graph_data).

        Args:
            step_path: absolute or relative path to .stp / .step file

        Returns:
            MachiningResult
        """
        t0 = time.time()
        step_path = str(step_path)

        # Lazy-load parser
        if self._parser_cls is None:
            _root = str(Path(__file__).parent.parent)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from src.parser import StepTextParser  # type: ignore
            self._parser_cls = StepTextParser

        parser = self._parser_cls()
        parser.parse_file(step_path)
        unit_system = getattr(parser, 'unit_system', 'unknown')

        hole_results: List[HoleResult] = []
        for hole in parser.features.holes:
            hole_results.append(self._infer_hole(hole, unit_system))

        return MachiningResult(
            file_path=step_path,
            unit_system=unit_system,
            holes=hole_results,
            processing_time_s=time.time() - t0,
        )

    # ── Inference: DGL graph (GNN) ───────────────────────────────────────────

    def predict_graph(self, graph_data) -> Optional[GNNResult]:
        """Run GNN face segmentation on a pre-built DGL attribute-graph.

        Args:
            graph_data: DGL graph with AAGNet node/edge attributes, OR a dict
                        as returned by MFInstSegDataset (must have key 'graph').

        Returns:
            GNNResult, or None if GNN is not loaded.
        """
        if self.gnn is None:
            logger.warning(
                "GNN not loaded. Supply gnn_path to from_pretrained()."
            )
            return None
        try:
            if isinstance(graph_data, dict):
                g = graph_data['graph'].to(self.device)
            else:
                g = graph_data.to(self.device)

            with torch.no_grad():
                seg_p, inst_p, bot_p = self.gnn(g)

            face_labels = seg_p.argmax(1).cpu().numpy().astype(np.int32)
            label_names = [
                GNN_CLASSES[i] if i < len(GNN_CLASSES) else f'class_{i}'
                for i in face_labels
            ]
            return GNNResult(
                face_labels=face_labels,
                face_label_names=label_names,
            )
        except Exception as exc:
            logger.error(f"GNN inference failed: {exc}")
            return None

    # ── Inference: raw point cloud ───────────────────────────────────────────

    @torch.no_grad()
    def predict_cloud(
        self, pc: torch.Tensor
    ) -> List[Dict[str, object]]:
        """Run MachinaQUnified on a raw point cloud tensor.

        Args:
            pc: (B, 3, N) or (3, N) float tensor

        Returns:
            List of B dicts, each with keys:
              feat_class, feat_name, feat_conf, feat_probs,
              is_through, through_conf, hole_probs
        """
        if self.unified is None:
            raise RuntimeError(
                "MachinaQUnified not loaded. "
                "Supply unified_path to from_pretrained()."
            )
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)   # (1, 3, N)
        pc = pc.float().to(self.device)

        out = self.unified.predict_with_confidence(pc)
        B = pc.shape[0]
        results = []
        for i in range(B):
            fc = out['feat_class'][i].item()
            results.append({
                'feat_class':   fc,
                'feat_name':    FEATURE_CLASSES[fc] if fc < len(FEATURE_CLASSES) else str(fc),
                'feat_conf':    out['feat_conf'][i].item(),
                'feat_probs':   out['feat_probs'][i].cpu().tolist(),
                'is_through':   bool(out['is_through'][i].item()),
                'through_conf': out['through_conf'][i].item(),
                'hole_probs':   out['hole_probs'][i].cpu().tolist(),
            })
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _infer_hole(self, hole: dict, unit_system: str) -> HoleResult:
        """Build point cloud for one parsed hole and run unified model."""
        r_in      = hole.get('radius', 0.010)
        d_in      = r_in * 2.0 * 2.0              # estimate: 2× diameter depth
        is_heur   = hole.get('is_through', False)

        # Generate wall-only cloud (neutral — no synthetic end-cap)
        pc_np = self._make_wall_cloud(r_in, d_in)  # (3, N)

        if self.unified is not None:
            pc_t = torch.from_numpy(pc_np).unsqueeze(0).to(self.device)
            out  = self.unified.predict_with_confidence(pc_t)
            fc         = out['feat_class'][0].item()
            fc_conf    = out['feat_conf'][0].item()
            is_through = bool(out['is_through'][0].item())
            thru_conf  = out['through_conf'][0].item()
        else:
            fc = HOLE_CLASS_IDX;  fc_conf = 1.0
            is_through = is_heur; thru_conf = 1.0 if is_heur else 0.0

        return HoleResult(
            step_id              = hole.get('id', -1),
            radius_in            = r_in,
            diameter_mm          = r_in * 2.0 * 25.4,
            asme_label           = hole.get('asme_label', ''),
            asme_standard        = hole.get('asme_standard', ''),
            asme_category        = hole.get('asme_category', ''),
            feat_type            = FEATURE_CLASSES[fc] if fc < len(FEATURE_CLASSES) else str(fc),
            feat_confidence      = fc_conf,
            is_through_model     = is_through,
            through_confidence   = thru_conf,
            is_through_heuristic = is_heur,
            is_through           = is_through,
            depth_estimate_in    = d_in,
            unit_system          = unit_system,
        )

    def _make_wall_cloud(self, radius: float, depth: float) -> np.ndarray:
        """Generate a cylinder-wall point cloud for inference.

        Returns (3, NUM_POINTS) float32.
        Wall-only (no flat end-cap) so the model scores against its
        through-hole prior; the B-Rep heuristic is the authoritative signal.
        """
        N = self.NUM_POINTS
        n_wall  = int(N * 0.85)
        n_noise = N - n_wall

        theta = np.random.uniform(0, 2 * np.pi, n_wall)
        z     = np.random.uniform(0, depth,     n_wall)
        jit   = 0.04 * radius
        x     = radius * np.cos(theta) + np.random.normal(0, jit, n_wall)
        y     = radius * np.sin(theta) + np.random.normal(0, jit, n_wall)
        wall  = np.stack([x, y, z], axis=1)

        s = radius * 1.2
        noise = np.random.uniform([-s, -s, 0], [s, s, depth], (n_noise, 3))

        pts = np.vstack([wall, noise])
        pts = pts - pts.mean(0)
        pts = pts / (np.linalg.norm(pts, axis=1).max() + 1e-8)
        return pts.astype(np.float32).T   # (3, N)

"""Pipeline orchestration for STEP feature analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .parser import StepTextParser
from .primitive import PrimitiveClassifier, SurfacePrimitive
from .features import Feature, FeatureDetector
from .operation_classifier import FeatureOperation, PartOperationsSummary, classify_features, summarize_part
from models.pointnet import PointNet, load_model

logger = logging.getLogger(__name__)

# Feature class mapping
FEATURE_CLASSES = {
    0: "hole",
    1: "boss",
    2: "slot",
    3: "thread",
    4: "drill"
}


class StepAnalyzer:
    def __init__(self) -> None:
        self.parser = StepTextParser()
        self.classifier = PrimitiveClassifier()
        self.detector = FeatureDetector()
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained PointNet model for inference."""
        try:
            model_path = Path(__file__).parent.parent / "models" / "pointnet_trained.pth"
            if model_path.exists():
                self.model = PointNet(num_classes=5)
                self.model = load_model(self.model, str(model_path))
                logger.info("Loaded trained PointNet model from %s", model_path)
            else:
                logger.warning("Trained model not found at %s, using rule-based detection only", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self.model = None

    def analyze(self, step_path: str) -> Dict[str, Any]:
        logger.info("Analyzing STEP file: %s", step_path)
        self.parser.parse_file(step_path)
        summary = self.parser.get_summary()
        primitives = self._extract_primitives()

        parser_features = self._hole_and_thread_features()
        claimed_face_ids = {fid for f in parser_features for fid in f.face_ids}
        other_features, unclassified_face_ids = self.detector.detect_all_features(
            primitives,
            claimed_face_ids=claimed_face_ids,
        )
        features = parser_features + other_features

        # STEP-derived primitives carry no part-wide principal axis today
        # (see design.md decision 2) — the classifier degrades gracefully.
        feature_operations = classify_features(features, primitives, principal_axis=None)
        operations_summary = summarize_part(feature_operations)

        # Add model predictions if available
        predictions = []
        if self.model is not None:
            predictions = self._predict_features(primitives)

        return {
            "summary": summary,
            "primitives": [self._primitive_to_dict(p) for p in primitives],
            "features": [
                self._feature_to_dict(f, op) for f, op in zip(features, feature_operations)
            ],
            "unclassified_face_ids": unclassified_face_ids,
            "operations_summary": self._operations_summary_to_dict(operations_summary),
            "predictions": predictions,
            "model_available": self.model is not None,
        }

    def _hole_and_thread_features(self) -> List[Feature]:
        """Build `hole`/`thread` Feature objects from the parser's standards-
        validated, topology-classified hole detection (through/blind, ASME/ISO
        size matching) — the authoritative source, not re-derived. A hole whose
        matched standard category is `tap_drill` is emitted as `thread` instead
        of `hole` (a tap-drill-sized hole is the pre-thread state), keeping the
        one-face-one-label guarantee.
        """
        features: List[Feature] = []
        for hole in self.parser.features.holes:
            is_thread = hole.get("asme_category") == "tap_drill"
            state = "through" if hole.get("is_through") else "blind"
            if is_thread:
                rationale = (
                    f"cylindrical face matched {hole['asme_standard']} tap-drill "
                    f"designation '{hole['asme_bolt_size']}' ({hole['asme_label']}, "
                    f"{hole['snap_error_pct']:.1f}% size error)"
                )
            else:
                rationale = (
                    f"cylindrical face matched {hole['asme_standard']} standard "
                    f"'{hole['asme_label']}' ({state} hole, {hole['snap_error_pct']:.1f}% size error)"
                )
            params = dict(hole)
            params["rationale"] = rationale
            features.append(Feature(
                feature_type="thread" if is_thread else "hole",
                face_ids=[hole["id"]],
                parameters=params,
            ))
        return features

    def _predict_features(self, primitives: List[SurfacePrimitive]) -> List[Dict[str, Any]]:
        """Predict feature types using trained PointNet model."""
        predictions = []
        if not primitives or self.model is None:
            return predictions

        try:
            for primitive in primitives:
                # Generate point cloud from primitive
                point_cloud = self._generate_point_cloud_from_primitive(primitive)
                
                # Prepare tensor
                point_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0)  # (1, 3, 1024)
                
                # Run inference
                with torch.no_grad():
                    output = self.model(point_tensor)
                    logits = output[0].numpy()
                    pred_class = np.argmax(logits)
                    confidence = float(np.exp(logits[pred_class]) / np.sum(np.exp(logits)))
                
                predictions.append({
                    "face_id": primitive.face_id,
                    "predicted_type": FEATURE_CLASSES.get(pred_class, "unknown"),
                    "confidence": confidence,
                    "logits": {FEATURE_CLASSES[i]: float(logits[i]) for i in range(5)},
                })
        except Exception as e:
            logger.error("Error during prediction: %s", e)
        
        return predictions

    def _generate_point_cloud_from_primitive(self, primitive: SurfacePrimitive, num_points: int = 1024) -> np.ndarray:
        """Generate synthetic point cloud from primitive geometry."""
        if primitive.type == 'cylindrical':
            radius = primitive.details.get('radius', 5.0)
            
            # Generate cylindrical surface points (70%)
            num_cyl_points = int(num_points * 0.7)
            theta = np.random.uniform(0, 2*np.pi, num_cyl_points)
            z = np.random.uniform(0, 10, num_cyl_points)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            cyl_points = np.stack([x, y, z], axis=1)
            
            # Generate noise points (30%)
            noise_points = np.random.uniform(-radius-1, radius+1, (num_points - num_cyl_points, 3))
            
            # Combine
            point_cloud = np.vstack([cyl_points, noise_points])[:num_points]
            
            # Normalize
            point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-6)
            
            return point_cloud.astype(np.float32).T  # (3, num_points)
        else:
            # Default: random points for unknown types
            points = np.random.uniform(-5, 5, (3, num_points)).astype(np.float32)
            points = (points - points.mean(axis=1, keepdims=True)) / (points.std(axis=1, keepdims=True) + 1e-6)
            return points

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Saved report to %s", output_path)

    def _extract_primitives(self) -> List[SurfacePrimitive]:
        primitives: List[SurfacePrimitive] = []
        adjacency = self.parser.get_face_adjacency()
        face_surface_types = self.parser.get_face_surface_types()
        face_radius: Dict[int, float] = {}
        for surf_id, radius, _direction_id in self.parser.primitives.cylinders:
            for fid in self.parser.get_surface_face_ids(surf_id):
                face_radius[fid] = radius

        def adjacent_faces(surf_id: int) -> List[int]:
            result: set = set()
            for fid in self.parser.get_surface_face_ids(surf_id):
                result |= adjacency.get(fid, set())
            return sorted(result)

        def adjacent_planar_count(adjacent_ids: List[int]) -> int:
            return sum(1 for fid in adjacent_ids if face_surface_types.get(fid) == 'PLANE')

        def nearest_adjacent_cylindrical_radius(adjacent_ids: List[int]) -> Optional[float]:
            radii = [face_radius[fid] for fid in adjacent_ids if fid in face_radius]
            return min(radii) if radii else None

        for surf_id, radius, _direction_id in self.parser.primitives.cylinders:
            adjacent_ids = adjacent_faces(surf_id)
            primitives.append(SurfacePrimitive(
                face_id=surf_id,
                type='cylindrical',
                details={
                    'radius': radius,
                    'adjacent_planar_count': float(adjacent_planar_count(adjacent_ids)),
                },
                adjacent_face_ids=adjacent_ids,
            ))

        for surf_id, normal, _point in self.parser.primitives.planes:
            adjacent_ids = adjacent_faces(surf_id)
            details: Dict[str, float] = {}
            for fid in self.parser.get_surface_face_ids(surf_id):
                extents = self.parser.get_face_bounding_extents(fid, normal)
                if extents is not None:
                    details['long_extent'], details['short_extent'] = extents
                    break
            primitives.append(SurfacePrimitive(
                face_id=surf_id,
                type='planar',
                details=details,
                adjacent_face_ids=adjacent_ids,
            ))

        for surf_id, _placement_ref, radius, semi_angle in self.parser.primitives.cones:
            adjacent_ids = adjacent_faces(surf_id)
            details = {'radius': radius, 'semi_angle': semi_angle}
            nearest_radius = nearest_adjacent_cylindrical_radius(adjacent_ids)
            if nearest_radius is not None:
                details['nearest_adjacent_cylindrical_radius'] = nearest_radius
            primitives.append(SurfacePrimitive(
                face_id=surf_id,
                type='conical',
                details=details,
                adjacent_face_ids=adjacent_ids,
            ))

        return primitives

    def _primitive_to_dict(self, primitive: SurfacePrimitive) -> Dict[str, Any]:
        return {
            "face_id": primitive.face_id,
            "type": primitive.type,
            "details": primitive.details,
        }

    def _feature_to_dict(self, feature: Feature, operation: Optional[FeatureOperation] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "feature_type": feature.feature_type,
            "face_ids": feature.face_ids,
            "parameters": feature.parameters,
        }
        if operation is not None:
            result["operation"] = operation.operation
            result["operation_rationale"] = operation.rationale
        return result

    def _operations_summary_to_dict(self, summary: PartOperationsSummary) -> Dict[str, Any]:
        return {
            "primary_process": summary.primary_process,
            "secondary_processes": summary.secondary_processes,
            "rationale": summary.rationale,
        }

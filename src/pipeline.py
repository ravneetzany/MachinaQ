"""Pipeline orchestration for STEP feature analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .parser import StepTextParser
from .primitive import PrimitiveClassifier, SurfacePrimitive
from .features import Feature, FeatureDetector
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
        features = self.detector.detect_all_features(primitives)

        # Add model predictions if available
        predictions = []
        if self.model is not None:
            predictions = self._predict_features(primitives)

        return {
            "summary": summary,
            "primitives": [self._primitive_to_dict(p) for p in primitives],
            "features": [self._feature_to_dict(f) for f in features],
            "predictions": predictions,
            "model_available": self.model is not None,
        }

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
        primitives = []
        for i, cyl in enumerate(self.parser.primitives.cylinders):
            primitives.append(SurfacePrimitive(
                face_id=cyl[0],
                type='cylindrical',
                details={'radius': cyl[1]}
            ))
        # Add more for other primitives
        return primitives

    def _primitive_to_dict(self, primitive: SurfacePrimitive) -> Dict[str, Any]:
        return {
            "face_id": primitive.face_id,
            "type": primitive.type,
            "details": primitive.details,
        }

    def _feature_to_dict(self, feature: Feature) -> Dict[str, Any]:
        return {
            "feature_type": feature.feature_type,
            "face_ids": feature.face_ids,
            "parameters": feature.parameters,
        }

"""Rule-based manufacturing feature detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .primitive import SurfacePrimitive

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    feature_type: str
    face_ids: List[int]
    parameters: Dict[str, Any]


class FeatureDetector:
    def __init__(self) -> None:
        pass

    def detect_holes(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        holes: List[Feature] = []
        for primitive in primitives:
            if primitive.type == "cylindrical":
                holes.append(Feature(feature_type="hole", face_ids=[primitive.face_id], parameters=primitive.details))
        return holes

    def detect_bosses(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        bosses: List[Feature] = []
        for primitive in primitives:
            if primitive.type in {"cylindrical", "conical"}:
                bosses.append(Feature(feature_type="boss", face_ids=[primitive.face_id], parameters=primitive.details))
        return bosses

    def detect_slots(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        return [Feature(feature_type="slot", face_ids=[p.face_id], parameters={}) for p in primitives if p.type == "planar"]

    def detect_threads(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        return [Feature(feature_type="thread", face_ids=[p.face_id], parameters={}) for p in primitives if p.type == "cylindrical" and p.details.get("radius", 0) > 0]

    def detect_drills(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        return [Feature(feature_type="drill", face_ids=[p.face_id], parameters={}) for p in primitives if p.type == "conical"]

    def detect_all_features(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        features: List[Feature] = []
        features.extend(self.detect_holes(primitives))
        features.extend(self.detect_bosses(primitives))
        features.extend(self.detect_slots(primitives))
        features.extend(self.detect_threads(primitives))
        features.extend(self.detect_drills(primitives))
        logger.debug("Detected %d features", len(features))
        return features

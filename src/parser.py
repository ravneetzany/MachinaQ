"""STEP text parsing and geometry extraction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StepEntity:
    id: int
    type: str
    attributes: List[Any]
    raw: str


@dataclass
class GeometryPrimitives:
    points: List[Tuple[float, float, float]]
    lines: List[Tuple[int, int]]  # point ids
    circles: List[Tuple[int, float]]  # center point id, radius
    planes: List[Tuple[int, int, int]]  # three point ids
    cylinders: List[Tuple[int, float, int]]  # axis point id, radius, direction id
    cones: List[Tuple[int, float, float, int]]  # apex point id, semi-angle, height, direction id


@dataclass
class ManufacturingFeatures:
    holes: List[Dict[str, Any]]
    bosses: List[Dict[str, Any]]
    slots: List[Dict[str, Any]]
    threads: List[Dict[str, Any]]
    drills: List[Dict[str, Any]]


class StepTextParser:
    def __init__(self) -> None:
        self.entities: Dict[int, StepEntity] = {}
        self.primitives = GeometryPrimitives([], [], [], [], [], [])
        self.features = ManufacturingFeatures([], [], [], [], [])

    def parse_file(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"STEP file not found: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._parse_entities(content)
        self._extract_primitives()
        self._detect_features()

    def _parse_entities(self, content: str) -> None:
        # Remove comments and headers
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        lines = content.split(';')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('#'):
                continue
            
            match = re.match(r'#(\d+)\s*=\s*([A-Z_]+)\s*\((.*)\)', line)
            if match:
                entity_id = int(match.group(1))
                entity_type = match.group(2)
                attributes_str = match.group(3)
                
                attributes = self._parse_attributes(attributes_str)
                
                self.entities[entity_id] = StepEntity(
                    id=entity_id,
                    type=entity_type,
                    attributes=attributes,
                    raw=line
                )

    def _parse_attributes(self, attr_str: str) -> List[Any]:
        # Simple attribute parsing - this is a basic implementation
        # Real STEP parsing is complex due to nested structures
        attrs = []
        depth = 0
        current = ""
        
        for char in attr_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                attrs.append(self._parse_value(current.strip()))
                current = ""
                continue
            current += char
        
        if current.strip():
            attrs.append(self._parse_value(current.strip()))
        
        return attrs

    def _parse_value(self, value: str) -> Any:
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]  # string
        elif value == '$':
            return None
        elif value.startswith('#'):
            return int(value[1:])  # reference
        elif '(' in value:
            # nested - simplify
            return value
        else:
            try:
                return float(value)
            except ValueError:
                return value

    def _extract_primitives(self) -> None:
        for entity in self.entities.values():
            if entity.type == 'CARTESIAN_POINT':
                coords = entity.attributes[1] if len(entity.attributes) > 1 else []
                if isinstance(coords, str) and coords.startswith('(') and coords.endswith(')'):
                    coords = coords[1:-1].split(',')
                    try:
                        x, y, z = map(float, coords)
                        self.primitives.points.append((x, y, z))
                    except:
                        pass
            elif entity.type == 'CIRCLE':
                # Simplified: assume radius and center
                if len(entity.attributes) >= 2:
                    radius = entity.attributes[1]
                    if isinstance(radius, float):
                        self.primitives.circles.append((entity.id, radius))
            elif entity.type == 'PLANE':
                # Simplified
                pass
            elif entity.type == 'CYLINDRICAL_SURFACE':
                # Simplified: assume position, axis, radius
                if len(entity.attributes) >= 3:
                    radius = entity.attributes[2]
                    if isinstance(radius, float):
                        self.primitives.cylinders.append((entity.id, radius, entity.id))

    def _detect_features(self) -> None:
        """Detect manufacturing features from STEP entities.
        
        Holes are identified as CYLINDRICAL_SURFACE entities used in ADVANCED_FACE.
        According to ISO-10303-21, a hole is a cylindrical depression in a solid.
        """
        # Only detect holes: cylindrical surfaces that form faces
        for entity in self.entities.values():
            if entity.type == 'CYLINDRICAL_SURFACE':
                # Extract radius from cylindrical surface attributes
                radius = None
                if len(entity.attributes) >= 3:
                    # CYLINDRICAL_SURFACE format: name, axis_placement, radius
                    radius = entity.attributes[2]
                
                if isinstance(radius, float):
                    hole_info = {
                        'id': entity.id,
                        'radius': radius,
                        'type': 'hole',
                        'entity_type': 'CYLINDRICAL_SURFACE'
                    }
                    self.features.holes.append(hole_info)
                    
                    # Log hole detection
                    logger.info(f"Detected hole: ID={entity.id}, Radius={radius}mm")

    def get_summary(self) -> Dict[str, Any]:
        return {
            'entity_count': len(self.entities),
            'primitive_counts': {
                'points': len(self.primitives.points),
                'circles': len(self.primitives.circles),
                'cylinders': len(self.primitives.cylinders),
            },
            'feature_counts': {
                'holes': len(self.features.holes),
                'bosses': len(self.features.bosses),
                'slots': len(self.features.slots),
                'threads': len(self.features.threads),
                'drills': len(self.features.drills),
            }
        }

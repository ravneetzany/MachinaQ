"""STEP text parsing and geometry extraction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .asme_standards import (
    # inch (ASME)
    classify_hole,
    PRACTICAL_MIN_RADIUS_IN,
    # metric (ISO)
    classify_hole_mm,
    PRACTICAL_MIN_RADIUS_MM,
    # shared
    DEFAULT_TOLERANCE,
)

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

        self.unit_system = self._detect_unit_system(content)
        logger.info(f"Unit system detected: {self.unit_system} ({path.name})")

        self._parse_entities(content)
        self._extract_primitives()
        self._detect_features()

    @staticmethod
    def _detect_unit_system(content: str) -> str:
        """Detect whether the STEP file geometry is in inches or millimetres.

        Strategy (in order of priority):
        1. Explicit CONVERSION_BASED_UNIT containing 'INCH' anywhere in file
           (NIST STC/FTC files encode "UNITS: INCHES" in PMI text annotations)
        2. SI_UNIT(.MILLI.,.METRE.) with no inch override → mm
        3. Numeric heuristic: median CYLINDRICAL_SURFACE radius.
           Values < 2.0  → inches (all NIST STC radii are 0.06–0.44)
           Values ≥ 2.0  → mm    (CTC radii are 5–50)

        Returns:
            'inch' or 'mm'
        """
        import re as _re

        upper = content.upper()

        # 1. Explicit INCH keyword anywhere — covers NIST STC PMI text
        if 'INCH' in upper:
            return 'inch'

        # 2. Explicit metric declaration
        if 'SI_UNIT(.MILLI.,.METRE.)' in upper or 'MILLIMETRE' in upper:
            # Also check for inline conversion factor 25.4 (inch→mm)
            if '25.4' in content or '0.0254' in content:
                return 'inch'
            return 'mm'

        # 3. Numeric heuristic on CYLINDRICAL_SURFACE radii
        nums = _re.findall(
            r'CYLINDRICAL_SURFACE\s*\([^,]+,[^,]+,([0-9]+\.?[0-9]*)\)', content
        )
        if nums:
            radii = [float(v) for v in nums]
            median = sorted(radii)[len(radii) // 2]
            return 'inch' if median < 2.0 else 'mm'

        return 'mm'  # safe default

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

    def _build_topology(self) -> tuple:
        """Build face→edges and edge→faces maps for through-hole detection.

        Returns:
            (face_edges, edge_to_faces, face_surface, surf_to_faces)
            face_edges     : {face_id -> set of edge_curve_ids}
            edge_to_faces  : {edge_curve_id -> set of face_ids}
            face_surface   : {face_id -> surface_type_str}
            surf_to_faces  : {surface_entity_id -> list of face_ids}
        """
        face_edges: Dict[int, set] = {}
        face_surface: Dict[int, str] = {}
        surf_of_face: Dict[int, int] = {}

        for fid, entity in self.entities.items():
            if entity.type != 'ADVANCED_FACE':
                continue
            edges: set = set()

            # ADVANCED_FACE: ('name', (bound_refs...), surface_ref, .T./.F.)
            # Bounds are inside the first (...) after the name string
            attrs = entity.attributes
            # attrs[0]=name, attrs[1]=bounds tuple (stored as raw str "(#b1,#b2,...)")
            # attrs[2]=surface ref, attrs[3]=orientation
            if len(attrs) >= 3:
                # Surface ref is attrs[2] (an int, from _parse_value)
                if isinstance(attrs[2], int):
                    sid = attrs[2]
                    surf_of_face[fid] = sid
                    if sid in self.entities:
                        face_surface[fid] = self.entities[sid].type

            # Traverse bounds -> edge_loops -> oriented_edges -> edge_curves
            # attrs[1] is stored as raw string "(#291)" from _parse_value nested case
            bounds_raw = attrs[1] if len(attrs) > 1 else ''
            if isinstance(bounds_raw, str):
                bound_refs = [int(r) for r in
                              __import__('re').findall(r'#(\d+)', bounds_raw)]
            elif isinstance(bounds_raw, int):
                bound_refs = [bounds_raw]
            else:
                bound_refs = []

            for bid in bound_refs:
                if bid not in self.entities:
                    continue
                bent = self.entities[bid]
                if bent.type not in ('FACE_OUTER_BOUND', 'FACE_BOUND'):
                    continue
                # FACE_OUTER_BOUND('', #edge_loop, .T.)
                for attr in bent.attributes:
                    if not isinstance(attr, int):
                        continue
                    if attr not in self.entities:
                        continue
                    loop_ent = self.entities[attr]
                    if loop_ent.type != 'EDGE_LOOP':
                        continue
                    # EDGE_LOOP('', (#oe1, #oe2, ...))
                    # oriented_edge refs are in attrs[1] as raw string
                    oe_raw = loop_ent.attributes[1] if len(loop_ent.attributes) > 1 else ''
                    if isinstance(oe_raw, str):
                        oe_refs = [int(r) for r in
                                   __import__('re').findall(r'#(\d+)', oe_raw)]
                    elif isinstance(oe_raw, int):
                        oe_refs = [oe_raw]
                    else:
                        oe_refs = []

                    for oeid in oe_refs:
                        if oeid not in self.entities:
                            continue
                        oe_ent = self.entities[oeid]
                        if oe_ent.type != 'ORIENTED_EDGE':
                            continue
                        # ORIENTED_EDGE('', *, *, #edge_curve, .T.)
                        # * wildcards are not parsed as #refs by _parse_value
                        # Only the edge_curve survives as an int attribute
                        for oa in oe_ent.attributes:
                            if isinstance(oa, int) and oa in self.entities:
                                if self.entities[oa].type == 'EDGE_CURVE':
                                    edges.add(oa)

            face_edges[fid] = edges

        # Reverse map
        edge_to_faces: Dict[int, set] = {}
        for fid, edges in face_edges.items():
            for ec in edges:
                edge_to_faces.setdefault(ec, set()).add(fid)

        # Group faces by surface entity
        surf_to_faces: Dict[int, list] = {}
        for fid, sid in surf_of_face.items():
            surf_to_faces.setdefault(sid, []).append(fid)

        return face_edges, edge_to_faces, face_surface, surf_to_faces

    def _classify_through_hole(
        self,
        surf_id: int,
        face_ids: list,
        face_edges: Dict[int, set],
        edge_to_faces: Dict[int, set],
        face_surface: Dict[int, str],
    ) -> bool:
        """Return True if the cylinder is a through hole, False if blind.

        Rules (in priority order):
        1. INCH unit files (NIST STC sheet-metal): always through by design.
        2. MM unit files — topology heuristic:
           - Count PLANE faces adjacent to the cylinder.
           - adj_planes == 0 → through (no flat caps found).
           - adj_planes == 1 → blind (one flat bottom cap).
           - adj_planes == 2 → through (entry + exit flat surfaces of a plate/wall).
           - adj_planes >= 3 → complex feature (counterbore/step); treat as through.
        """
        # Rule 1 — sheet metal (inch) files are all through holes
        if getattr(self, 'unit_system', 'mm') == 'inch':
            return True

        # Rule 2 — topology for mm files
        cyl_ec: set = set()
        for fid in face_ids:
            cyl_ec |= face_edges.get(fid, set())

        adj: set = set()
        for ec in cyl_ec:
            adj |= edge_to_faces.get(ec, set())
        adj -= set(face_ids)

        adj_planes = sum(1 for f in adj if face_surface.get(f) == 'PLANE')

        if adj_planes == 1:
            return False   # one flat bottom cap → blind
        return True        # 0, 2, or more → through (both ends open)

    def _detect_features(self) -> None:
        """Detect manufacturing features from STEP entities.

        Holes are identified as CYLINDRICAL_SURFACE entities used in ADVANCED_FACE.
        According to ISO-10303-21, a hole is a cylindrical depression in a solid.

        UNITS: determined by :meth:`_detect_unit_system` which reads the STEP
        header. NIST STC files are in inches (ASME B18/B94 lookup). CTC, FTC,
        and holeTrain files are in millimetres (ISO 273 / DIN 338 lookup).

        Through-hole detection: builds a full B-Rep topology graph
        (face→edges→faces) and checks whether a PLANE face caps one end.
        STC (inch) files are always through by sheet-metal definition.
        """
        use_metric  = getattr(self, 'unit_system', 'mm') == 'mm'
        classify_fn = classify_hole_mm   if use_metric else classify_hole
        min_radius  = PRACTICAL_MIN_RADIUS_MM if use_metric else PRACTICAL_MIN_RADIUS_IN
        unit_label  = 'mm' if use_metric else 'in'

        # Build topology graph for through-hole detection
        face_edges, edge_to_faces, face_surface, surf_to_faces = self._build_topology()

        skipped_small    = 0
        skipped_nonstand = 0

        for entity in self.entities.values():
            if entity.type != 'CYLINDRICAL_SURFACE':
                continue

            # CYLINDRICAL_SURFACE format: name, axis_placement, radius
            radius = None
            if len(entity.attributes) >= 3:
                radius = entity.attributes[2]

            if not isinstance(radius, float):
                continue

            # ── 1. Reject sub-practical holes ─────────────────────────────
            if radius < min_radius:
                skipped_small += 1
                logger.debug(
                    f"Skipping sub-standard hole: ID={entity.id}, "
                    f"R={radius:.5f}{unit_label} (dia={radius*2:.5f}{unit_label}) — "
                    f"below practical min {min_radius:.4f}{unit_label}"
                )
                continue

            # ── 2. Validate against ASME (inch) or ISO (mm) standards ─────
            info = classify_fn(radius, tolerance=DEFAULT_TOLERANCE)

            if not info['matched']:
                skipped_nonstand += 1
                logger.debug(
                    f"Skipping non-standard hole: ID={entity.id}, "
                    f"R={radius:.5f}{unit_label} (dia={radius*2:.5f}{unit_label}) — "
                    f"nearest standard ±{info['error_pct']:.1f}%"
                )
                continue

            # ── 3. Through-hole topology check ────────────────────────────
            face_ids = surf_to_faces.get(entity.id, [])
            is_through = self._classify_through_hole(
                entity.id, face_ids, face_edges, edge_to_faces, face_surface
            )

            # ── 4. Accept — store with standard + topology classification ──
            dia_mm = info['diameter_mm']
            hole_info = {
                'id':            entity.id,
                'radius':        info['snap_radius'],   # snapped to standard
                'raw_radius':    radius,
                'unit_system':   unit_label,
                'diameter_mm':   dia_mm,
                'is_through':    is_through,
                'type':          'hole',
                'entity_type':   'CYLINDRICAL_SURFACE',
                # Standard classification fields
                'asme_label':    info['label'],
                'asme_standard': info['standard'],
                'asme_category': info['category'],
                'asme_bolt_size':info['bolt_size'],
                'asme_fit':      info['fit'],
                'snap_error_pct':info['error_pct'],
            }
            self.features.holes.append(hole_info)

            hole_type = 'THROUGH' if is_through else 'BLIND'
            logger.info(
                f"Detected {hole_type} hole: ID={entity.id}, "
                f"R={radius:.5f}{unit_label} -> {info['label']} "
                f"[dia={dia_mm:.4f}mm, err={info['error_pct']:.1f}%]"
            )

        if skipped_small or skipped_nonstand:
            logger.info(
                f"Hole filter ({unit_label}): accepted={len(self.features.holes)}, "
                f"skipped_small={skipped_small}, "
                f"skipped_non_standard={skipped_nonstand}"
            )

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

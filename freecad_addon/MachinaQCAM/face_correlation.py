"""Best-effort, position-based correlation between a FreeCAD-selected face
and MachinaQ-reported primitives.

Per design.md decision 3, this is an approximate heuristic (nearest point
for planar primitives, nearest point on the axis line for cylindrical/
conical primitives) — never an exact face-id correspondence.

The distance math (`point_distance`, `point_line_distance`, `nearest_primitive`)
is pure Python with no FreeCAD dependency, so it's directly unit-testable;
only `correlate_faces` touches actual FreeCAD `Face` objects.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

Vector3 = Tuple[float, float, float]

_AXIS_POINT_KEYS = ("axis_px", "axis_py", "axis_pz")
_AXIS_DIR_KEYS = ("axis_dx", "axis_dy", "axis_dz")


def point_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def point_line_distance(point: Vector3, line_point: Vector3, line_direction: Vector3) -> float:
    """Perpendicular distance from `point` to the infinite line defined by
    `line_point` + t * `line_direction`."""
    norm = math.sqrt(sum(c * c for c in line_direction)) or 1.0
    d = tuple(c / norm for c in line_direction)
    to_point = tuple(point[i] - line_point[i] for i in range(3))
    dot = sum(to_point[i] * d[i] for i in range(3))
    projection = tuple(to_point[i] - dot * d[i] for i in range(3))
    return math.sqrt(sum(c * c for c in projection))


def _primitive_position(details: Dict[str, Any]) -> Optional[Tuple[Vector3, Optional[Vector3]]]:
    if not all(k in details for k in _AXIS_POINT_KEYS):
        return None
    point = (details["axis_px"], details["axis_py"], details["axis_pz"])
    direction = None
    if all(k in details for k in _AXIS_DIR_KEYS):
        direction = (details["axis_dx"], details["axis_dy"], details["axis_dz"])
    return point, direction


def nearest_primitive(
    face_center: Vector3, primitives: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find the primitive (from a MachinaQ report's `primitives` list)
    nearest to `face_center`, using point-distance for planar primitives
    and axis-line distance for cylindrical/conical ones. Returns a copy of
    the matched primitive dict with an added `distance` key, or None if no
    primitive in the list carries resolvable position data."""
    best: Optional[Dict[str, Any]] = None
    best_distance: Optional[float] = None

    for primitive in primitives:
        details = primitive.get("details", {})
        position = _primitive_position(details)
        if position is None:
            continue
        point, direction = position

        if primitive.get("type") == "planar" or direction is None:
            distance = point_distance(face_center, point)
        else:
            distance = point_line_distance(face_center, point, direction)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = primitive

    if best is None:
        return None
    result = dict(best)
    result["distance"] = best_distance
    return result


def correlate_faces(faces: List[Any], primitives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """faces: FreeCAD Face objects (must expose `.CenterOfMass`).
    Returns one entry per face: {'center': (x,y,z), 'match': primitive-or-None}.
    Always an approximate, position-based match — see module docstring."""
    results = []
    for face in faces:
        com = face.CenterOfMass
        center = (com.x, com.y, com.z)
        results.append({"center": center, "match": nearest_primitive(center, primitives)})
    return results

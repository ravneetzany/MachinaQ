"""Best-effort, position-*and-orientation*-based correlation between a
FreeCAD-selected face and MachinaQ-reported primitives.

Per design.md decision 3, this is an approximate heuristic — never an
exact face-id correspondence. The original version matched on position
alone (nearest point for planar primitives, nearest point on the axis
line for cylindrical/conical primitives), which real-world testing showed
mismatches on parts with several similar, closely-spaced features (a
"bolt pattern" of near-identical holes/slots at different positions
around a part) — exactly the risk design.md's Risks section already
flagged as a known limitation.

**Improvement**: when the selected face's own orientation (surface normal
for planar faces, axis direction for cylindrical/conical faces) is
available, it's combined with position into a single score — a primitive
whose own axis/normal direction doesn't match the selected face's is
penalized even if positionally close, which disambiguates the common case
of several similar features arranged around a part at different rotational
positions (each with a different real-world orientation) far better than
position alone. Falls back to pure position-based matching when
orientation isn't available (e.g. curved/exotic surface types this module
doesn't know how to extract an axis from) — so this is a strict
enhancement over the original behavior, not a replacement requirement.

The scoring math (`point_distance`, `point_line_distance`,
`direction_similarity`, `nearest_primitive`) is pure Python with no
FreeCAD dependency, so it's directly unit-testable; only `correlate_faces`
and `_face_axis` touch actual FreeCAD `Face`/`Surface` objects.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

Vector3 = Tuple[float, float, float]

_AXIS_POINT_KEYS = ("axis_px", "axis_py", "axis_pz")
_AXIS_DIR_KEYS = ("axis_dx", "axis_dy", "axis_dz")

#: How strongly a direction mismatch penalizes an otherwise-close match, in
#: the same units as `distance` (STEP export units, typically mm). A
#: mismatch of 90° (`direction_similarity` = 0) adds this much to the
#: score; a perfect match (`direction_similarity` = 1) adds nothing. Chosen
#: to be comparable to a typical small-feature spacing, so direction
#: dominates over minor distance differences without swamping a
#: genuinely-much-closer primitive of a different orientation entirely.
DIRECTION_MISMATCH_PENALTY = 20.0


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


def direction_similarity(a: Vector3, b: Vector3) -> float:
    """Cosine similarity between two directions, in [0, 1] — 1 = parallel
    (or anti-parallel; axis direction sign is arbitrary in this codebase,
    so `abs()` treats the two as equivalent), 0 = perpendicular."""
    norm_a = math.sqrt(sum(c * c for c in a)) or 1.0
    norm_b = math.sqrt(sum(c * c for c in b)) or 1.0
    dot = sum(a[i] * b[i] for i in range(3)) / (norm_a * norm_b)
    return min(1.0, abs(dot))


def _primitive_position(details: Dict[str, Any]) -> Optional[Tuple[Vector3, Optional[Vector3]]]:
    if not all(k in details for k in _AXIS_POINT_KEYS):
        return None
    point = (details["axis_px"], details["axis_py"], details["axis_pz"])
    direction = None
    if all(k in details for k in _AXIS_DIR_KEYS):
        direction = (details["axis_dx"], details["axis_dy"], details["axis_dz"])
    return point, direction


def nearest_primitive(
    face_center: Vector3,
    primitives: List[Dict[str, Any]],
    face_direction: Optional[Vector3] = None,
) -> Optional[Dict[str, Any]]:
    """Find the primitive (from a MachinaQ report's `primitives` list)
    nearest to `face_center`, using point-distance for planar primitives
    and axis-line distance for cylindrical/conical ones.

    When `face_direction` (the selected face's own normal/axis) is given,
    a primitive whose stored direction doesn't align with it is penalized
    via `DIRECTION_MISMATCH_PENALTY`, disambiguating similar features at
    different positions/orientations far better than distance alone (see
    module docstring). Without it, matching is position-only, as before.

    Returns a copy of the matched primitive dict with `distance` and (when
    `face_direction` was usable) `direction_similarity` keys added, or None
    if no primitive in the list carries resolvable position data."""
    best: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_distance: Optional[float] = None
    best_similarity: Optional[float] = None

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

        similarity = None
        score = distance
        if face_direction is not None and direction is not None:
            similarity = direction_similarity(face_direction, direction)
            score = distance + DIRECTION_MISMATCH_PENALTY * (1.0 - similarity)

        if best_score is None or score < best_score:
            best_score = score
            best_distance = distance
            best_similarity = similarity
            best = primitive

    if best is None:
        return None
    result = dict(best)
    result["distance"] = best_distance
    if best_similarity is not None:
        result["direction_similarity"] = best_similarity
    return result


def _face_axis(face: Any) -> Optional[Vector3]:
    """Best-effort extraction of a FreeCAD Face's own orientation: the
    surface normal for a planar face, the axis direction for a
    cylindrical/conical one. Returns None for surface types this doesn't
    recognize (e.g. B-spline/free-form) rather than guessing — matching
    then falls back to position-only, per the module docstring."""
    try:
        surface = face.Surface
        axis = getattr(surface, "Axis", None)
        if axis is not None:
            return (axis.x, axis.y, axis.z)
    except Exception:
        pass
    return None


def correlate_faces(faces: List[Any], primitives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """faces: FreeCAD Face objects (must expose `.CenterOfMass`; `.Surface`
    is used opportunistically for orientation, when extractable).
    Returns one entry per face: {'center': (x,y,z), 'match': primitive-or-None}.
    Always an approximate, position-and-orientation-based match — see
    module docstring."""
    results = []
    for face in faces:
        com = face.CenterOfMass
        center = (com.x, com.y, com.z)
        direction = _face_axis(face)
        results.append({
            "center": center,
            "match": nearest_primitive(center, primitives, face_direction=direction),
        })
    return results

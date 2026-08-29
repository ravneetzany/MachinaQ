"""Shared axis/vector geometry used by parametric-source ingestion and operation classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]

DEFAULT_ANGLE_TOL_DEG = 5.0
DEFAULT_DIST_TOL = 1e-3


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def normalize(a: Vector3) -> Vector3:
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


@dataclass(frozen=True)
class Axis:
    """A directed line in 3D space: a unit direction plus a point on the line."""

    direction: Vector3
    point: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", normalize(self.direction))


def angle_between_deg(d1: Vector3, d2: Vector3) -> float:
    """Angle between two directions, in [0, 90] degrees (direction sign-agnostic)."""
    d1n, d2n = normalize(d1), normalize(d2)
    cos_theta = abs(_dot(d1n, d2n))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def point_line_distance(point: Vector3, axis: Axis) -> float:
    """Perpendicular distance from `point` to the infinite line defined by `axis`."""
    to_point = _sub(point, axis.point)
    cross = _cross(to_point, axis.direction)
    return _norm(cross)


def is_coaxial(
    a: Axis,
    b: Axis,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    dist_tol: float = DEFAULT_DIST_TOL,
) -> bool:
    """True when two axes share direction (within angle_tol_deg) and lie on the
    same line (within dist_tol of each other's line)."""
    if angle_between_deg(a.direction, b.direction) > angle_tol_deg:
        return False
    return point_line_distance(a.point, b) <= dist_tol


def is_axis_aligned_with_any(
    direction: Vector3,
    candidates: Tuple[Vector3, Vector3, Vector3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
) -> bool:
    """True when `direction` is parallel (within tolerance) to one of the given
    orthogonal machine axes (default: global X/Y/Z)."""
    return any(angle_between_deg(direction, c) <= angle_tol_deg for c in candidates)


# Flattened storage keys for embedding an Axis inside a `Dict[str, float]`
# (e.g. `SurfacePrimitive.details`), since that dict is float-valued only.
_AXIS_KEYS = ("axis_dx", "axis_dy", "axis_dz", "axis_px", "axis_py", "axis_pz")


def axis_to_details(axis: Axis) -> dict:
    """Flatten an Axis into float-valued keys suitable for `SurfacePrimitive.details`."""
    dx, dy, dz = axis.direction
    px, py, pz = axis.point
    return {"axis_dx": dx, "axis_dy": dy, "axis_dz": dz, "axis_px": px, "axis_py": py, "axis_pz": pz}


def orthonormal_basis(normal: Vector3) -> Tuple[Vector3, Vector3]:
    """Return two orthonormal vectors (u, v) spanning the plane perpendicular
    to `normal`, for projecting 3D points into that plane's 2D coordinates."""
    n = normalize(normal)
    helper: Vector3 = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = normalize(_cross(n, helper))
    v = normalize(_cross(n, u))
    return u, v


def bounding_extents_2d(
    points: list, origin: Vector3, u: Vector3, v: Vector3
) -> Tuple[float, float]:
    """Project 3D `points` onto the (u, v) plane through `origin` and return
    the (extent_u, extent_v) bounding-box size in that plane."""
    if not points:
        return (0.0, 0.0)
    us = [_dot(_sub(p, origin), u) for p in points]
    vs = [_dot(_sub(p, origin), v) for p in points]
    return (max(us) - min(us), max(vs) - min(vs))


def axis_from_details(details: dict) -> Optional[Axis]:
    """Recover an Axis previously stored via `axis_to_details`, or None if absent."""
    if not all(k in details for k in _AXIS_KEYS):
        return None
    return Axis(
        direction=(details["axis_dx"], details["axis_dy"], details["axis_dz"]),
        point=(details["axis_px"], details["axis_py"], details["axis_pz"]),
    )

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "freecad_addon" / "MachinaQCAM"))

from face_correlation import (  # noqa: E402
    correlate_faces,
    direction_similarity,
    nearest_primitive,
    point_distance,
    point_line_distance,
)


def test_point_distance() -> None:
    assert point_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0


def test_point_line_distance_perpendicular_offset() -> None:
    # Line along Z through origin; point offset by 5 in X at some Z height.
    distance = point_line_distance((5.0, 0.0, 42.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert abs(distance - 5.0) < 1e-9


def test_point_line_distance_on_the_line_is_zero() -> None:
    distance = point_line_distance((0.0, 0.0, 100.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert abs(distance) < 1e-9


def test_nearest_primitive_planar_uses_point_distance() -> None:
    primitives = [
        {"face_id": 1, "type": "planar", "details": {"axis_px": 0.0, "axis_py": 0.0, "axis_pz": 0.0}},
        {"face_id": 2, "type": "planar", "details": {"axis_px": 10.0, "axis_py": 0.0, "axis_pz": 0.0}},
    ]
    match = nearest_primitive((1.0, 0.0, 0.0), primitives)
    assert match["face_id"] == 1


def test_nearest_primitive_cylindrical_uses_axis_line_distance() -> None:
    primitives = [
        {
            "face_id": 1, "type": "cylindrical",
            "details": {
                "axis_px": 0.0, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 0.0, "axis_dy": 0.0, "axis_dz": 1.0,
            },
        },
        {
            "face_id": 2, "type": "cylindrical",
            "details": {
                "axis_px": 100.0, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 0.0, "axis_dy": 0.0, "axis_dz": 1.0,
            },
        },
    ]
    # A face far along Z on the first cylinder's wall (near its axis line) should match primitive 1
    match = nearest_primitive((3.0, 0.0, 500.0), primitives)
    assert match["face_id"] == 1
    assert match["distance"] == 3.0


def test_nearest_primitive_returns_none_when_no_position_data() -> None:
    primitives = [{"face_id": 1, "type": "planar", "details": {}}]
    assert nearest_primitive((0.0, 0.0, 0.0), primitives) is None


def test_direction_similarity_parallel_is_one() -> None:
    assert abs(direction_similarity((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) - 1.0) < 1e-9


def test_direction_similarity_perpendicular_is_zero() -> None:
    assert abs(direction_similarity((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))) < 1e-9


def test_direction_similarity_antiparallel_is_one() -> None:
    # Axis direction sign is arbitrary in this codebase; anti-parallel axes
    # represent the same real-world orientation.
    assert abs(direction_similarity((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)) - 1.0) < 1e-9


def test_nearest_primitive_disambiguates_similar_features_by_orientation() -> None:
    # Two similar planar features close together — pure position-based
    # matching picks the marginally closer one (primitive 2), which happens
    # to be wrongly oriented relative to the selected face; primitive 1 is
    # the true match, matching the selected face's own normal.
    primitives = [
        {
            "face_id": 1, "type": "planar",
            "details": {
                "axis_px": 1.0, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 0.0, "axis_dy": 0.0, "axis_dz": 1.0,
            },
        },
        {
            "face_id": 2, "type": "planar",
            "details": {
                "axis_px": 0.9, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 1.0, "axis_dy": 0.0, "axis_dz": 0.0,
            },
        },
    ]
    face_center = (0.0, 0.0, 0.0)
    face_direction = (0.0, 0.0, 1.0)

    # Without orientation info, the slightly-closer (but wrongly oriented)
    # primitive 2 wins.
    assert nearest_primitive(face_center, primitives)["face_id"] == 2

    # With the selected face's own orientation, primitive 1 (matching normal)
    # wins instead.
    match = nearest_primitive(face_center, primitives, face_direction=face_direction)
    assert match["face_id"] == 1
    assert abs(match["direction_similarity"] - 1.0) < 1e-9


class _FakeAxis:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class _FakeSurface:
    def __init__(self, axis: _FakeAxis) -> None:
        self.Axis = axis


class _FakeFace:
    def __init__(self, center: _FakeAxis, axis: _FakeAxis) -> None:
        self.CenterOfMass = center
        self.Surface = axis and _FakeSurface(axis)


def test_correlate_faces_uses_face_surface_orientation() -> None:
    primitives = [
        {
            "face_id": 1, "type": "planar",
            "details": {
                "axis_px": 1.0, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 0.0, "axis_dy": 0.0, "axis_dz": 1.0,
            },
        },
        {
            "face_id": 2, "type": "planar",
            "details": {
                "axis_px": 0.9, "axis_py": 0.0, "axis_pz": 0.0,
                "axis_dx": 1.0, "axis_dy": 0.0, "axis_dz": 0.0,
            },
        },
    ]
    face = _FakeFace(_FakeAxis(0.0, 0.0, 0.0), _FakeAxis(0.0, 0.0, 1.0))
    results = correlate_faces([face], primitives)
    assert len(results) == 1
    assert results[0]["match"]["face_id"] == 1


def test_correlate_faces_falls_back_to_position_when_no_surface_axis() -> None:
    primitives = [
        {"face_id": 1, "type": "planar", "details": {"axis_px": 0.0, "axis_py": 0.0, "axis_pz": 0.0}},
        {"face_id": 2, "type": "planar", "details": {"axis_px": 10.0, "axis_py": 0.0, "axis_pz": 0.0}},
    ]
    face = _FakeFace(_FakeAxis(1.0, 0.0, 0.0), None)
    results = correlate_faces([face], primitives)
    assert results[0]["match"]["face_id"] == 1


def test_nearest_primitive_uses_nearest_boundary_point_not_just_centroid() -> None:
    # A freeform primitive whose vertex-average centroid is far from the
    # selected face, but one of its boundary_points is close — the large
    # curved-patch case that surfaced this (see face_correlation.py's
    # module docstring, second improvement).
    primitives = [
        {
            "face_id": 1, "type": "freeform",
            "details": {
                "axis_px": 100.0, "axis_py": 100.0, "axis_pz": 0.0,  # far centroid
                "boundary_points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            },
        },
        {
            "face_id": 2, "type": "planar",
            "details": {"axis_px": 50.0, "axis_py": 50.0, "axis_pz": 0.0},
        },
    ]
    match = nearest_primitive((0.5, 0.5, 0.0), primitives)
    assert match["face_id"] == 1
    assert match["distance"] < 1.0  # nearest boundary point, not the far centroid (~141mm away)


def test_nearest_primitive_without_boundary_points_is_unaffected() -> None:
    primitives = [
        {"face_id": 1, "type": "planar", "details": {"axis_px": 0.0, "axis_py": 0.0, "axis_pz": 0.0}},
    ]
    match = nearest_primitive((1.0, 0.0, 0.0), primitives)
    assert match["face_id"] == 1
    assert match["distance"] == 1.0

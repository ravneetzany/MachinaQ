import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "freecad_addon" / "MachinaQCAM"))

from face_correlation import nearest_primitive, point_distance, point_line_distance  # noqa: E402


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

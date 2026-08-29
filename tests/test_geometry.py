from src.geometry import (
    Axis,
    axis_from_details,
    axis_to_details,
    is_axis_aligned_with_any,
    is_coaxial,
)


def test_coaxial_axes_within_tolerance() -> None:
    a = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    b = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 5.0))
    assert is_coaxial(a, b) is True


def test_non_coaxial_parallel_offset_axes() -> None:
    a = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    b = Axis(direction=(0.0, 0.0, 1.0), point=(5.0, 0.0, 0.0))
    assert is_coaxial(a, b) is False


def test_non_coaxial_perpendicular_axes() -> None:
    a = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    b = Axis(direction=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0))
    assert is_coaxial(a, b) is False


def test_axis_details_roundtrip() -> None:
    axis = Axis(direction=(0.0, 1.0, 0.0), point=(1.0, 2.0, 3.0))
    details = axis_to_details(axis)
    recovered = axis_from_details(details)
    assert recovered is not None
    assert is_coaxial(axis, recovered)


def test_axis_from_details_missing_keys() -> None:
    assert axis_from_details({}) is None


def test_axis_aligned_with_orthogonal_machine_axis() -> None:
    assert is_axis_aligned_with_any((0.0, 0.0, 1.0)) is True
    assert is_axis_aligned_with_any((1.0, 1.0, 1.0)) is False

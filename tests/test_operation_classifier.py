from src.features import Feature
from src.geometry import Axis
from src.operation_classifier import (
    Operation,
    classify_feature,
    classify_features,
    summarize_part,
)
from src.primitive import SurfacePrimitive


def test_operation_constants_exist() -> None:
    assert set(Operation.ALL) == {
        "turning",
        "drilling",
        "face_milling",
        "3_axis_milling",
        "5_axis_milling",
        "unknown",
    }


PRINCIPAL_AXIS = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))


def _cyl_primitive(face_id: int, axis: Axis, radius: float = 5.0) -> SurfacePrimitive:
    from src.geometry import axis_to_details

    details = {"radius": radius}
    details.update(axis_to_details(axis))
    return SurfacePrimitive(face_id=face_id, type="cylindrical", details=details)


def _planar_primitive(face_id: int, axis: Axis) -> SurfacePrimitive:
    from src.geometry import axis_to_details

    details = axis_to_details(axis)
    return SurfacePrimitive(face_id=face_id, type="planar", details=details)


def test_cylindrical_coaxial_with_principal_axis_is_turning() -> None:
    prim = _cyl_primitive(1, PRINCIPAL_AXIS)
    feature = Feature(feature_type="boss", face_ids=[1], parameters={})
    result = classify_feature(feature, {1: prim}, PRINCIPAL_AXIS)
    assert result.operation == Operation.TURNING
    assert "coaxial" in result.rationale


def test_hole_not_coaxial_but_orthogonal_is_drilling() -> None:
    hole_axis = Axis(direction=(1.0, 0.0, 0.0), point=(0.0, 3.0, 0.0))  # perpendicular to part axis, X-aligned
    prim = _cyl_primitive(2, hole_axis)
    feature = Feature(feature_type="hole", face_ids=[2], parameters={})
    result = classify_feature(feature, {2: prim}, PRINCIPAL_AXIS)
    assert result.operation == Operation.DRILLING


def test_planar_feature_on_prismatic_part_is_face_or_3axis_milling() -> None:
    # No principal_axis at all -> prismatic-part scenario per spec
    prim = SurfacePrimitive(face_id=3, type="planar", details={})
    feature = Feature(feature_type="slot", face_ids=[3], parameters={})
    by_face = {3: prim}
    result = classify_feature(feature, by_face, None)
    assert result.operation == Operation.THREE_AXIS_MILLING


def test_non_orthogonal_non_coaxial_feature_is_5_axis_milling() -> None:
    slanted_axis = Axis(direction=(1.0, 1.0, 1.0), point=(0.0, 0.0, 0.0))
    prim = _planar_primitive(4, slanted_axis)
    feature = Feature(feature_type="slot", face_ids=[4], parameters={})
    result = classify_feature(feature, {4: prim}, PRINCIPAL_AXIS)
    assert result.operation == Operation.FIVE_AXIS_MILLING


def test_unknown_primitive_type_yields_unknown_operation() -> None:
    prim = SurfacePrimitive(face_id=5, type="unknown", details={})
    feature = Feature(feature_type="unclassified", face_ids=[5], parameters={})
    result = classify_feature(feature, {5: prim}, PRINCIPAL_AXIS)
    assert result.operation == Operation.UNKNOWN
    assert "insufficient" in result.rationale


def test_step_path_fallback_without_axis_does_not_error() -> None:
    prim = SurfacePrimitive(face_id=6, type="cylindrical", details={"radius": 3.0})
    feature = Feature(feature_type="hole", face_ids=[6], parameters={})
    result = classify_feature(feature, {6: prim}, None)
    assert result.operation == Operation.DRILLING
    assert result.rationale


def test_summary_single_operation_has_no_secondary() -> None:
    prim = _cyl_primitive(1, PRINCIPAL_AXIS)
    feature = Feature(feature_type="boss", face_ids=[1], parameters={})
    ops = classify_features([feature], [prim], PRINCIPAL_AXIS)
    summary = summarize_part(ops)
    assert summary.primary_process == Operation.TURNING
    assert summary.secondary_processes == []


def test_summary_turned_body_with_noncoaxial_secondary_feature() -> None:
    turned_prim = _cyl_primitive(1, PRINCIPAL_AXIS)
    slanted_axis = Axis(direction=(1.0, 1.0, 1.0), point=(0.0, 0.0, 0.0))
    keyway_prim = _planar_primitive(2, slanted_axis)

    body_feature = Feature(feature_type="boss", face_ids=[1], parameters={})
    keyway_feature = Feature(feature_type="slot", face_ids=[2], parameters={})

    ops = classify_features(
        [body_feature, keyway_feature], [turned_prim, keyway_prim], PRINCIPAL_AXIS
    )
    summary = summarize_part(ops)
    assert summary.primary_process == Operation.TURNING
    assert Operation.FIVE_AXIS_MILLING in summary.secondary_processes
    assert "cannot be produced by" in summary.rationale


def test_summary_excludes_unknown_from_vote() -> None:
    unknown_prim = SurfacePrimitive(face_id=1, type="unknown", details={})
    unknown_feature = Feature(feature_type="unclassified", face_ids=[1], parameters={})
    ops = classify_features([unknown_feature], [unknown_prim], PRINCIPAL_AXIS)
    summary = summarize_part(ops)
    assert summary.primary_process == Operation.UNKNOWN
    assert summary.secondary_processes == []


def _toroidal_primitive(face_id: int, axis: Axis) -> SurfacePrimitive:
    from src.geometry import axis_to_details

    details = {"major_radius": 5.0, "minor_radius": 1.0}
    details.update(axis_to_details(axis))
    return SurfacePrimitive(face_id=face_id, type="toroidal", details=details)


def test_axis_aligned_toroidal_face_is_3_axis_milling() -> None:
    axis_aligned = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    prim = _toroidal_primitive(1, axis_aligned)
    feature = Feature(feature_type="elongated_boss", face_ids=[1], parameters={})
    result = classify_feature(feature, {1: prim}, None)
    assert result.operation == Operation.THREE_AXIS_MILLING


def test_non_axis_aligned_toroidal_face_is_5_axis_milling() -> None:
    slanted = Axis(direction=(1.0, 1.0, 1.0), point=(0.0, 0.0, 0.0))
    prim = _toroidal_primitive(1, slanted)
    feature = Feature(feature_type="elongated_boss", face_ids=[1], parameters={})
    result = classify_feature(feature, {1: prim}, None)
    assert result.operation == Operation.FIVE_AXIS_MILLING


def test_freeform_face_is_always_5_axis_milling() -> None:
    prim = SurfacePrimitive(face_id=1, type="freeform", details={"long_extent": 30.0, "short_extent": 5.0})
    feature = Feature(feature_type="elongated_boss", face_ids=[1], parameters={})
    result = classify_feature(feature, {1: prim}, None)
    assert result.operation == Operation.FIVE_AXIS_MILLING
    assert "no resolvable axis" in result.rationale

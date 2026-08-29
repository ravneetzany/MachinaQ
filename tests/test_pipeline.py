from pathlib import Path

from src.geometry import axis_from_details
from src.pipeline import StepAnalyzer

HOLE_FIXTURE = "nist_sfa/holeTrain/HoleData04.step"
CYLINDER_FIXTURE = "nist_sfa/nist_ctc_01_asme1_ap242-e1.stp"
OPERATION_CHECKPOINT = Path("outputs/machinaq_operation_classifier.pth")


def test_analyzer_instantiates() -> None:
    analyzer = StepAnalyzer()
    assert analyzer is not None


def test_analyzer_instantiates_regardless_of_operation_checkpoint_presence() -> None:
    # Exercises both branches of _load_operation_model() without mocking the
    # filesystem — whichever state the checkpoint happens to be in locally,
    # instantiation must not raise either way.
    analyzer = StepAnalyzer()
    if OPERATION_CHECKPOINT.exists():
        assert analyzer.operation_model is not None
    else:
        assert analyzer.operation_model is None


def test_analyze_missing_step_file() -> None:
    analyzer = StepAnalyzer()
    try:
        analyzer.analyze("missing_file.step")
    except FileNotFoundError:
        assert True
    except RuntimeError:
        assert True
    else:
        assert False, "Expected missing file failure"


def test_hole_features_carry_through_blind_and_standard_fields() -> None:
    analyzer = StepAnalyzer()
    report = analyzer.analyze(HOLE_FIXTURE)
    hole_features = [f for f in report["features"] if f["feature_type"] == "hole"]
    assert hole_features
    params = hole_features[0]["parameters"]
    assert "is_through" in params
    assert "asme_label" in params
    assert params["rationale"]


def test_step_derived_cylindrical_primitives_carry_a_resolved_axis() -> None:
    analyzer = StepAnalyzer()
    report = analyzer.analyze(CYLINDER_FIXTURE)
    cylindrical = [p for p in report["primitives"] if p["type"] == "cylindrical"]
    assert cylindrical
    axis = axis_from_details(cylindrical[0]["details"])
    assert axis is not None


def test_report_includes_unclassified_face_ids() -> None:
    analyzer = StepAnalyzer()
    report = analyzer.analyze(HOLE_FIXTURE)
    assert "unclassified_face_ids" in report


def test_report_includes_per_feature_operation_and_operations_summary() -> None:
    analyzer = StepAnalyzer()
    report = analyzer.analyze(HOLE_FIXTURE)
    assert report["features"]
    for feature in report["features"]:
        assert "operation" in feature
        assert "operation_rationale" in feature
    assert "operations_summary" in report
    summary = report["operations_summary"]
    assert "primary_process" in summary
    assert "secondary_processes" in summary
    assert "rationale" in summary


def test_operation_predictions_absent_without_checkpoint() -> None:
    analyzer = StepAnalyzer()
    analyzer.operation_model = None  # force the "no checkpoint" branch deterministically
    report = analyzer.analyze(HOLE_FIXTURE)
    assert "operation_predictions" not in report
    assert "operation" in report["features"][0]  # existing rule-derived fields unchanged
    assert "operations_summary" in report


def test_operation_predictions_present_with_checkpoint() -> None:
    from models.operation_classifier_net import OperationClassifierNet

    analyzer = StepAnalyzer()
    analyzer.operation_model = OperationClassifierNet()  # freshly-initialized is enough to exercise the path
    report = analyzer.analyze(HOLE_FIXTURE)
    assert "operation_predictions" in report
    for pred in report["operation_predictions"]:
        assert "predicted_operation" in pred
        assert "confidence" in pred
    # existing rule-derived fields still present and unchanged in shape
    assert "operation" in report["features"][0]
    assert "operations_summary" in report


def test_large_planar_faces_are_classified_and_operated_on() -> None:
    """A large planar face (e.g. a stock top face) must reach operation
    classification as a `planar_face` feature, not be silently left in
    `unclassified_face_ids` — see add-face-milling-feature-detection."""
    analyzer = StepAnalyzer()
    report = analyzer.analyze(HOLE_FIXTURE)

    face_features = [f for f in report["features"] if f["feature_type"] == "planar_face"]
    assert face_features
    for feature in face_features:
        assert feature["operation"] != "unknown"
        assert feature["face_ids"][0] not in report["unclassified_face_ids"]


def test_toroidal_primitive_round_trips_axis() -> None:
    content = (
        "#50 = CARTESIAN_POINT('',(1.0,2.0,3.0));\n"
        "#51 = DIRECTION('',(0.0,0.0,1.0));\n"
        "#52 = DIRECTION('',(1.0,0.0,0.0));\n"
        "#53 = AXIS2_PLACEMENT_3D('',#50,#51,#52);\n"
        "#60 = TOROIDAL_SURFACE('',#53,5.0,1.0);\n"
    )
    analyzer = StepAnalyzer()
    analyzer.parser._parse_entities(content)
    analyzer.parser._extract_primitives()
    primitives = analyzer._extract_primitives()
    toroidal = [p for p in primitives if p.type == "toroidal"]
    assert len(toroidal) == 1
    assert toroidal[0].details["major_radius"] == 5.0
    assert toroidal[0].details["minor_radius"] == 1.0
    axis = axis_from_details(toroidal[0].details)
    assert axis is not None
    assert axis.point == (1.0, 2.0, 3.0)


_SQUARE_FACE_STEP = (
    "#1 = CARTESIAN_POINT('',(0.0,0.0,0.0));\n"
    "#2 = CARTESIAN_POINT('',(10.0,0.0,0.0));\n"
    "#3 = CARTESIAN_POINT('',(10.0,4.0,0.0));\n"
    "#4 = CARTESIAN_POINT('',(0.0,4.0,0.0));\n"
    "#11 = VERTEX_POINT('',#1);\n"
    "#12 = VERTEX_POINT('',#2);\n"
    "#13 = VERTEX_POINT('',#3);\n"
    "#14 = VERTEX_POINT('',#4);\n"
    "#21 = EDGE_CURVE('',#11,#12,#100,.T.);\n"
    "#22 = EDGE_CURVE('',#12,#13,#100,.T.);\n"
    "#23 = EDGE_CURVE('',#13,#14,#100,.T.);\n"
    "#24 = EDGE_CURVE('',#14,#11,#100,.T.);\n"
    "#31 = ORIENTED_EDGE('',*,*,#21,.T.);\n"
    "#32 = ORIENTED_EDGE('',*,*,#22,.T.);\n"
    "#33 = ORIENTED_EDGE('',*,*,#23,.T.);\n"
    "#34 = ORIENTED_EDGE('',*,*,#24,.T.);\n"
    "#40 = EDGE_LOOP('',(#31,#32,#33,#34));\n"
    "#41 = FACE_OUTER_BOUND('',#40,.T.);\n"
)


def test_freeform_primitive_carries_extents_when_resolvable() -> None:
    content = _SQUARE_FACE_STEP + (
        "#60 = B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#100,#101)),.UNSPECIFIED.,.F.,.F.,.F.);\n"
        "#70 = ADVANCED_FACE('',(#41),#60,.T.);\n"
    )
    analyzer = StepAnalyzer()
    analyzer.parser._parse_entities(content)
    analyzer.parser._extract_primitives()
    primitives = analyzer._extract_primitives()
    freeform = [p for p in primitives if p.type == "freeform"]
    assert len(freeform) == 1
    assert freeform[0].face_id == 60
    assert abs(freeform[0].details["long_extent"] - 10.0) < 1e-6
    assert abs(freeform[0].details["short_extent"] - 4.0) < 1e-6


def test_freeform_primitive_empty_details_when_unresolvable() -> None:
    content = (
        "#60 = B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#100,#101)),.UNSPECIFIED.,.F.,.F.,.F.);\n"
    )  # no ADVANCED_FACE references #60 at all -> no resolvable face vertices
    analyzer = StepAnalyzer()
    analyzer.parser._parse_entities(content)
    analyzer.parser._extract_primitives()
    primitives = analyzer._extract_primitives()
    freeform = [p for p in primitives if p.type == "freeform"]
    assert len(freeform) == 1
    assert freeform[0].details == {}


def test_no_face_appears_in_more_than_one_feature_type() -> None:
    """Regression: end-to-end analysis on a real NIST file must not double-label a face."""
    analyzer = StepAnalyzer()
    report = analyzer.analyze(HOLE_FIXTURE)

    face_to_types: dict = {}
    for feature in report["features"]:
        for face_id in feature["face_ids"]:
            face_to_types.setdefault(face_id, set()).add(feature["feature_type"])

    duplicated = {fid: types for fid, types in face_to_types.items() if len(types) > 1}
    assert duplicated == {}

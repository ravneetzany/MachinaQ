from src.pipeline import StepAnalyzer

HOLE_FIXTURE = "nist_sfa/holeTrain/HoleData04.step"


def test_analyzer_instantiates() -> None:
    analyzer = StepAnalyzer()
    assert analyzer is not None


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

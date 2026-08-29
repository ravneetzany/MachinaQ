from pathlib import Path

from src.pipeline import StepAnalyzer

HOLE_FIXTURE = "nist_sfa/holeTrain/HoleData04.step"
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

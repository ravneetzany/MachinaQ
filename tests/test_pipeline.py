from src.pipeline import StepAnalyzer


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

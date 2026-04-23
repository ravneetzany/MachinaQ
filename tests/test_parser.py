import pytest

from src.parser import StepParser


def test_parser_imports() -> None:
    parser = StepParser()
    assert parser is not None


def test_parser_summary_fails_for_missing_file() -> None:
    parser = StepParser()
    with pytest.raises(FileNotFoundError):
        parser.summary("missing_file.step")

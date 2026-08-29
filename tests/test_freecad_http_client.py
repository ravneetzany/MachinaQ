import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "freecad_addon" / "MachinaQCAM"))

from http_client import MachinaQErrorResponse, MachinaQUnreachableError, classify  # noqa: E402


def test_classify_raises_unreachable_on_connection_failure() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(MachinaQUnreachableError):
            classify("some.step", api_url="http://127.0.0.1:9999")


def test_classify_raises_error_response_on_http_error() -> None:
    body = json.dumps({"detail": "STEP file not found"}).encode("utf-8")

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:8000/analyze", code=404, msg="Not Found",
            hdrs=None, fp=None,
        )

    http_error = urllib.error.HTTPError(
        url="http://127.0.0.1:8000/analyze", code=404, msg="Not Found", hdrs=None, fp=None,
    )
    http_error.read = lambda: body

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(MachinaQErrorResponse) as exc_info:
            classify("missing.step")
        assert exc_info.value.status_code == 404
        assert "STEP file not found" in exc_info.value.detail


def test_classify_returns_parsed_json_on_success() -> None:
    response_body = json.dumps({"features": [], "operations_summary": {}}).encode("utf-8")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    with patch("urllib.request.urlopen", return_value=_FakeResponse()):
        result = classify("part.step")
        assert result == {"features": [], "operations_summary": {}}

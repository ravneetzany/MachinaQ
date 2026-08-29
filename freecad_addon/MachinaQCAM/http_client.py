"""HTTP client for MachinaQ's /analyze endpoint, used by the FreeCAD command.

Pure Python, no FreeCAD-specific imports, so this module is testable under
MachinaQ's own .venv as well as inside FreeCAD's embedded Python (which may
or may not have `requests` installed — falls back to the standard library's
`urllib.request` when it doesn't).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 30.0


class MachinaQUnreachableError(Exception):
    """The MachinaQ API could not be reached at all (connection failure,
    DNS failure, timeout before any response was received)."""


class MachinaQErrorResponse(Exception):
    """The MachinaQ API responded, but with a non-success status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"MachinaQ API returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _post_json_requests(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import requests

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise MachinaQUnreachableError(str(exc)) from exc

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise MachinaQErrorResponse(response.status_code, detail)

    return response.json()


def _post_json_urllib(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise MachinaQErrorResponse(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise MachinaQUnreachableError(str(exc.reason)) from exc
    except OSError as exc:
        raise MachinaQUnreachableError(str(exc)) from exc


def classify(
    step_path: str,
    api_url: str = DEFAULT_API_URL,
    output_path: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """POST a STEP file path to MachinaQ's /analyze endpoint and return the
    parsed JSON report.

    Raises:
        MachinaQUnreachableError: the API could not be reached at all.
        MachinaQErrorResponse: the API responded with a non-success status.
    """
    url = api_url.rstrip("/") + "/analyze"
    payload: Dict[str, Any] = {"step_path": step_path}
    if output_path:
        payload["output"] = output_path

    try:
        import requests  # noqa: F401
        has_requests = True
    except ImportError:
        has_requests = False

    if has_requests:
        return _post_json_requests(url, payload, timeout)
    return _post_json_urllib(url, payload, timeout)

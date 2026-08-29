"""MachinaQ API URL preference for the FreeCAD addon.

Uses FreeCAD's own preferences store when running inside FreeCAD (so the
user can change it via Edit > Preferences), falling back to the plain
default when FreeCAD isn't available (e.g. under a plain Python test run).
"""

from __future__ import annotations

# Plain (not relative) import: FreeCAD adds this Mod/MachinaQCAM/ directory
# itself to sys.path, so sibling modules are imported flat, not as a package.
from http_client import DEFAULT_API_URL

_PREF_GROUP = "User parameter:BaseApp/Preferences/Mod/MachinaQCAM"
_PREF_KEY = "ApiUrl"


def get_api_url() -> str:
    try:
        import FreeCAD

        params = FreeCAD.ParamGet(_PREF_GROUP)
        return params.GetString(_PREF_KEY, DEFAULT_API_URL) or DEFAULT_API_URL
    except ImportError:
        return DEFAULT_API_URL


def set_api_url(url: str) -> None:
    import FreeCAD

    params = FreeCAD.ParamGet(_PREF_GROUP)
    params.SetString(_PREF_KEY, url)

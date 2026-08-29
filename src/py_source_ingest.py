"""Static ingestion of FreeCAD wrapper-script (.py) part definitions.

Uses Python's `ast` module to statically find `make_*(...)` part-constructor
calls and resolve their literal/keyword arguments via `ast.literal_eval`,
without importing the script (these scripts import `FreeCAD` at module
scope, which requires FreeCAD's bundled Python) or executing FreeCAD. Each
recognized wrapper function maps to a small, explicit geometry template
(see design.md decision 3) since the real FreeCAD-generated `Shape` isn't
computed — these are approximations, not exact geometry.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .geometry import Axis, axis_to_details, reduce_principal_axis
from .primitive import SurfacePrimitive


class PySourceUnsupported(Exception):
    """Raised when a file has no recognized `make_*` call, or a call's
    arguments/wrapper function can't be statically resolved."""


@dataclass
class PySourceIngestResult:
    primitives: List[SurfacePrimitive]
    principal_axis: Optional[Axis]


def _make_axisymmetric_primitive(radius: float) -> SurfacePrimitive:
    axis = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    details: Dict[str, float] = {"radius": radius}
    details.update(axis_to_details(axis))
    return SurfacePrimitive(face_id=1, type="cylindrical", details=details)


_METRIC_DIAMETER_RE = re.compile(r"^M(\d+(?:\.\d+)?)$", re.IGNORECASE)
#: Used when a fastener's `diameter` argument isn't a parseable "M<n>"
#: designation (e.g. the default "Auto", or an imperial size) — a
#: documented approximation, not a real lookup.
_DEFAULT_FASTENER_RADIUS_MM = 4.0

#: A small, well-known ISO 15 / DIN 625 deep-groove ball-bearing
#: bore/OD series (mirrors freecad-parts-library/lib/standards/
#: bearings_iso15.py's published values) — a geometry template for
#: known designations, not a re-parse of that file.
_BEARING_BORE_OD_MM = {
    "608": (8, 22), "6000": (10, 26), "6001": (12, 28), "6002": (15, 32),
    "6003": (17, 35), "6004": (20, 42), "6005": (25, 47),
    "6200": (10, 30), "6201": (12, 32), "6202": (15, 35), "6203": (17, 40),
    "6204": (20, 47), "6205": (25, 52),
    "6300": (10, 35), "6301": (12, 37), "6302": (15, 42), "6303": (17, 47),
    "6304": (20, 52), "6305": (25, 62),
}


def _template_make_fastener(args: Dict[str, Any]) -> List[SurfacePrimitive]:
    diameter = args.get("diameter", "Auto")
    radius = _DEFAULT_FASTENER_RADIUS_MM
    if isinstance(diameter, str):
        match = _METRIC_DIAMETER_RE.match(diameter.strip())
        if match:
            radius = float(match.group(1)) / 2.0
    return [_make_axisymmetric_primitive(radius)]


def _template_make_gear(args: Dict[str, Any]) -> List[SurfacePrimitive]:
    module = args.get("module")
    if not isinstance(module, (int, float)):
        raise PySourceUnsupported("make_gear: 'module' argument is not a resolvable number")
    num_teeth = args.get("num_teeth")
    if isinstance(num_teeth, (int, float)):
        radius = (module * num_teeth) / 2.0
    else:
        radius = module * 10.0  # placeholder pitch radius when tooth count is unknown
    return [_make_axisymmetric_primitive(radius)]


def _template_make_compression_spring(args: Dict[str, Any]) -> List[SurfacePrimitive]:
    coil_od = args.get("coil_od")
    if not isinstance(coil_od, (int, float)):
        raise PySourceUnsupported("make_compression_spring: 'coil_od' argument is not a resolvable number")
    return [_make_axisymmetric_primitive(coil_od / 2.0)]


def _template_make_ball_bearing(args: Dict[str, Any]) -> List[SurfacePrimitive]:
    designation = args.get("designation")
    if not isinstance(designation, str) or designation not in _BEARING_BORE_OD_MM:
        raise PySourceUnsupported(f"make_ball_bearing: unrecognized designation {designation!r}")
    _bore, od = _BEARING_BORE_OD_MM[designation]
    return [_make_axisymmetric_primitive(od / 2.0)]


#: Positional-argument names for each recognized wrapper, matching its real
#: signature, so a positional ast.Call argument can be mapped to the right
#: keyword before evaluation.
_WRAPPER_PARAM_NAMES: Dict[str, List[str]] = {
    "make_fastener": ["fastener_type", "diameter", "length", "doc", "name"],
    "make_gear": ["gear_type", "module", "num_teeth", "height", "bore",
                  "mate_teeth", "shaft_angle", "doc"],
    "make_compression_spring": ["wire_d", "coil_od", "pitch", "n_coils"],
    "make_ball_bearing": ["designation", "ball_count", "separate"],
}

_WRAPPER_TEMPLATES: Dict[str, Callable[[Dict[str, Any]], List[SurfacePrimitive]]] = {
    "make_fastener": _template_make_fastener,
    "make_gear": _template_make_gear,
    "make_compression_spring": _template_make_compression_spring,
    "make_ball_bearing": _template_make_ball_bearing,
}


def _find_make_calls(tree: ast.AST) -> List[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("make_")
        ):
            calls.append(node)
    return calls


def _resolve_call_args(call: ast.Call, param_names: List[str]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for i, arg_node in enumerate(call.args):
        if i >= len(param_names):
            raise PySourceUnsupported("more positional arguments than known parameter names")
        try:
            resolved[param_names[i]] = ast.literal_eval(arg_node)
        except (ValueError, TypeError):
            raise PySourceUnsupported(f"positional argument {i} is not statically resolvable")
    for kw in call.keywords:
        if kw.arg is None:
            raise PySourceUnsupported("cannot statically resolve **kwargs unpacking")
        try:
            resolved[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            raise PySourceUnsupported(f"keyword argument '{kw.arg}' is not statically resolvable")
    return resolved


def ingest_source(source: str) -> PySourceIngestResult:
    """Parse `.py` source text, find `make_*(...)` calls, and return the
    extracted primitives plus a reduced principal axis, without importing
    the script or requiring a FreeCAD installation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PySourceUnsupported(f"source does not parse as Python: {exc}")

    calls = _find_make_calls(tree)
    if not calls:
        raise PySourceUnsupported("no make_*(...) part-constructor call found")

    all_primitives: List[SurfacePrimitive] = []
    for call in calls:
        func_name = call.func.id  # type: ignore[union-attr]
        if func_name not in _WRAPPER_TEMPLATES:
            raise PySourceUnsupported(f"unrecognized wrapper function '{func_name}'")
        param_names = _WRAPPER_PARAM_NAMES[func_name]
        args = _resolve_call_args(call, param_names)
        all_primitives.extend(_WRAPPER_TEMPLATES[func_name](args))

    for i, primitive in enumerate(all_primitives, start=1):
        primitive.face_id = i

    principal_axis = reduce_principal_axis(all_primitives)
    return PySourceIngestResult(primitives=all_primitives, principal_axis=principal_axis)


def ingest_file(path: Union[str, Path]) -> PySourceIngestResult:
    source = Path(path).read_text(encoding="utf-8")
    return ingest_source(source)


_EXCLUDED_DIR_NAMES = {"lib"}


def discover_py_files(root: Union[str, Path]) -> List[Path]:
    """Find `.py` part files under `root` that define a recognized `make_*`
    call, excluding shared/support files under `lib/` (e.g. `lib/common.py`)."""
    root = Path(root)
    results: List[Path] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if any(part in _EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts[:-1]):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        if _find_make_calls(tree):
            results.append(path)
    return results


@dataclass
class PySourceDiscoveryEntry:
    path: Path
    result: Optional[PySourceIngestResult]
    error: Optional[str]


def ingest_directory(root: Union[str, Path]) -> List[PySourceDiscoveryEntry]:
    """Ingest every discovered `.py` part file under `root`. A file that
    fails to parse is reported with an error reason rather than raising."""
    entries: List[PySourceDiscoveryEntry] = []
    for path in discover_py_files(root):
        try:
            result = ingest_file(path)
            entries.append(PySourceDiscoveryEntry(path=path, result=result, error=None))
        except PySourceUnsupported as exc:
            entries.append(PySourceDiscoveryEntry(path=path, result=None, error=str(exc)))
    return entries

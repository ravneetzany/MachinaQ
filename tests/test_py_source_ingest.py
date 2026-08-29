import sys

import pytest

from src.py_source_ingest import (
    PySourceUnsupported,
    discover_py_files,
    ingest_directory,
    ingest_file,
    ingest_source,
)

LIB_ROOT = "/home/ravneetzany/projects/freecad-parts-library"
FASTENER_PATH = f"{LIB_ROOT}/fasteners/fastener.py"


def test_fastener_call_resolved_without_importing_freecad() -> None:
    assert "FreeCAD" not in sys.modules
    result = ingest_file(FASTENER_PATH)
    assert "FreeCAD" not in sys.modules
    assert len(result.primitives) == 1
    assert result.primitives[0].type == "cylindrical"
    assert result.primitives[0].details["radius"] == 4.0  # M8 -> r=4.0mm


def test_gear_wrapper_template() -> None:
    result = ingest_source('make_gear("InvoluteGear", module=2.0, num_teeth=24, height=8, bore=8)')
    assert len(result.primitives) == 1
    assert result.primitives[0].type == "cylindrical"
    assert result.primitives[0].details["radius"] == pytest.approx(24.0)  # (2.0*24)/2


def test_compression_spring_wrapper_template() -> None:
    result = ingest_source("make_compression_spring(wire_d=2, coil_od=20, pitch=4, n_coils=8)")
    assert result.primitives[0].details["radius"] == 10.0


def test_ball_bearing_wrapper_template() -> None:
    result = ingest_source('make_ball_bearing("6204")')
    assert result.primitives[0].details["radius"] == 23.5  # od=47 -> r=23.5


def test_all_wrapper_calls_are_axisymmetric() -> None:
    for source in [
        'make_fastener("ISO4014", diameter="M8", length="30")',
        'make_gear("InvoluteGear", module=2.0, num_teeth=24)',
        "make_compression_spring(wire_d=2, coil_od=20, pitch=4, n_coils=8)",
        'make_ball_bearing("6204")',
    ]:
        result = ingest_source(source)
        assert result.principal_axis is not None


def test_non_resolvable_call_reports_unparsed_not_exception() -> None:
    with pytest.raises(PySourceUnsupported):
        ingest_source("make_gear('InvoluteGear', module=compute_module())")


def test_discovery_excludes_lib_directory() -> None:
    files = discover_py_files(LIB_ROOT)
    assert files
    names = {f.name for f in files}
    assert "common.py" not in names
    assert "bearings_iso15.py" not in names


def test_directory_ingest_reports_no_unhandled_exception() -> None:
    entries = ingest_directory(LIB_ROOT)
    assert entries
    assert any(e.error is None for e in entries)
    for e in entries:
        assert (e.result is None) != (e.error is None)

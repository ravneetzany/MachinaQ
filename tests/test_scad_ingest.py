from src.scad_ingest import ingest_source, ingest_file
from src.geometry import is_coaxial

LIB_ROOT = "/home/ravneetzany/projects/openscad-parts-library"

BUSHING_PATH = f"{LIB_ROOT}/bearings/bushing.scad"
L_BRACKET_PATH = f"{LIB_ROOT}/milled_parts/l_bracket.scad"
STEPPED_SHAFT_PATH = f"{LIB_ROOT}/transmission/stepped_shaft.scad"


def test_bushing_extracts_expected_cylindrical_primitives() -> None:
    result = ingest_file(BUSHING_PATH)
    cyl = [p for p in result.primitives if p.type == "cylindrical"]
    assert len(cyl) == 4
    radii = sorted(p.details["radius"] for p in cyl)
    # od=16 -> r=8, id=10 -> r=5 (x2, tube bore + flange bore), flange_od=22 -> r=11
    assert radii == [5.0, 5.0, 8.0, 11.0]


def test_stepped_shaft_extracts_expected_cylindrical_primitives() -> None:
    result = ingest_file(STEPPED_SHAFT_PATH)
    cyl = [p for p in result.primitives if p.type == "cylindrical"]
    assert len(cyl) == 3
    radii = sorted(p.details["radius"] for p in cyl)
    # segments = [[20,40],[16,30],[12,20]] -> diameters 20/16/12 -> radii 10/8/6
    assert radii == [6.0, 8.0, 10.0]


def test_bushing_reduces_to_a_single_principal_axis() -> None:
    result = ingest_file(BUSHING_PATH)
    assert result.principal_axis is not None
    assert is_coaxial(result.principal_axis, result.principal_axis)


def test_l_bracket_has_no_single_principal_axis() -> None:
    result = ingest_file(L_BRACKET_PATH)
    assert result.principal_axis is None
    # both plates' own primitives are still returned, with their own transforms
    planar = [p for p in result.primitives if p.type == "planar"]
    assert len(planar) == 2
    normals = {(round(p.details["axis_dx"]), round(p.details["axis_dy"]), round(p.details["axis_dz"])) for p in planar}
    assert len(normals) == 2  # the two plates face different directions


def test_stepped_shaft_keeps_single_axis_despite_milled_keyway() -> None:
    result = ingest_file(STEPPED_SHAFT_PATH)
    assert result.principal_axis is not None
    planar = [p for p in result.primitives if p.type == "planar"]
    assert len(planar) == 1  # the keyway cut


def test_unsupported_construct_raises_reportable_error() -> None:
    import pytest
    from src.scad_ingest import ScadUnsupported

    with pytest.raises(ScadUnsupported):
        ingest_source("hull() { sphere(5); }")


def test_discovery_excludes_lib_directory() -> None:
    from src.scad_ingest import discover_scad_files

    files = discover_scad_files(LIB_ROOT)
    assert files  # at least one file found
    assert all("lib" not in f.relative_to(LIB_ROOT).parts[:-1] for f in files)


def test_ingested_primitives_compatible_with_existing_feature_detector() -> None:
    """Ingested .scad primitives are SurfacePrimitive-shaped, so the existing
    FeatureDetector accepts them and returns the same (features, unclassified)
    shape it returns for STEP-derived primitives."""
    from src.features import FeatureDetector

    result = ingest_file(BUSHING_PATH)
    features, unclassified_face_ids = FeatureDetector().detect_all_features(result.primitives)
    assert isinstance(features, list)
    assert isinstance(unclassified_face_ids, list)


def test_directory_ingest_reports_unparsed_without_raising() -> None:
    from src.scad_ingest import ingest_directory

    entries = ingest_directory(LIB_ROOT)
    assert entries
    assert any(e.error is None for e in entries)  # at least one parsed
    for e in entries:
        assert (e.result is None) != (e.error is None)  # exactly one is set

from pathlib import Path

from src.geometry import Axis, axis_to_details
from src.operation_classifier_dataset import (
    VECTOR_DIM,
    build_corpus,
    build_freecad_examples,
    build_scad_examples,
    build_step_examples,
    vectorize,
)
from src.primitive import SurfacePrimitive

NIST_HOLETRAIN = list(Path("nist_sfa/holeTrain").glob("*.step"))
SCAD_LIB = "/home/ravneetzany/projects/openscad-parts-library"
FREECAD_LIB = "/home/ravneetzany/projects/freecad-parts-library"


def test_vectorize_is_deterministic() -> None:
    axis = Axis(direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0))
    details = {"radius": 5.0}
    details.update(axis_to_details(axis))
    prim = SurfacePrimitive(face_id=1, type="cylindrical", details=details)

    v1 = vectorize(prim, axis)
    v2 = vectorize(prim, axis)
    assert v1 == v2
    assert len(v1) == VECTOR_DIM


def test_vectorize_planar_no_axis() -> None:
    prim = SurfacePrimitive(face_id=1, type="planar", details={})
    vector = vectorize(prim, None)
    assert vector == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_build_step_examples_from_holetrain() -> None:
    assert NIST_HOLETRAIN
    examples = build_step_examples(NIST_HOLETRAIN)
    assert examples
    assert all(len(ex.vector) == VECTOR_DIM for ex in examples)


def test_build_step_examples_skips_broken_path_without_raising() -> None:
    examples = build_step_examples(["nonexistent_file.step"])
    assert examples == []


def test_build_scad_examples() -> None:
    examples = build_scad_examples(SCAD_LIB)
    assert examples
    assert all(len(ex.vector) == VECTOR_DIM for ex in examples)


def test_build_freecad_examples() -> None:
    examples = build_freecad_examples(FREECAD_LIB)
    assert examples
    assert all(len(ex.vector) == VECTOR_DIM for ex in examples)


def test_step_sourced_examples_are_labeled_by_the_rules_directly() -> None:
    """STEP labels come straight from StepAnalyzer.analyze()'s own
    `operation` field — i.e. exactly what classify_feature() would produce,
    since that's what computed it in the first place."""
    from src.pipeline import StepAnalyzer

    path = str(NIST_HOLETRAIN[0])
    report = StepAnalyzer().analyze(path)
    expected_ops = {f["operation"] for f in report["features"] if "operation" in f}

    examples = build_step_examples([path])
    found_ops = {ex.label for ex in examples}
    assert found_ops <= expected_ops | found_ops  # every found label was among the report's own operations
    assert found_ops == expected_ops


def test_scad_examples_use_the_ingesters_own_principal_axis() -> None:
    """.scad-sourced examples must reflect classify_feature() run with the
    ingester's actual principal_axis (not None) — verified indirectly via
    a coaxial cylindrical part (bushing.scad) producing a `turning` label,
    which only classify_with_axis's coaxial branch can produce."""
    examples = build_scad_examples(SCAD_LIB)
    bushing_examples = [ex for ex in examples if ex.source == "bushing.scad"]
    assert bushing_examples
    assert any(ex.label == "turning" for ex in bushing_examples)


def test_build_corpus_combines_all_three_sources() -> None:
    corpus = build_corpus(NIST_HOLETRAIN, SCAD_LIB, FREECAD_LIB)
    sources = {ex.source for ex in corpus}
    assert any(s.endswith(".step") for s in sources)
    assert any(s.endswith(".scad") for s in sources)
    assert any(s.endswith(".py") for s in sources)

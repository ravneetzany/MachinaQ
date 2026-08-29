"""Assemble (feature_vector, label) training examples for the learned
operation classifier by running the *existing* rule-based
`operation_classifier.classify_feature()` over a multi-source geometry
corpus (NIST STEP files, the OpenSCAD parts library, the FreeCAD wrapper-
script parts library).

Every label here comes from the rule engine itself — this is a self-
distillation setup, not independent ground truth. See design.md's
"Correction found during implementation" notes in the related changes
for context on why.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import py_source_ingest, scad_ingest
from .features import Feature
from .geometry import Axis, axis_from_details, is_axis_aligned_with_any, is_coaxial
from .operation_classifier import classify_features
from .primitive import SurfacePrimitive

logger = logging.getLogger(__name__)

VECTOR_DIM = 8


@dataclass
class TrainingExample:
    vector: Tuple[float, ...]
    label: str
    source: str


def vectorize(primitive: SurfacePrimitive, principal_axis: Optional[Axis]) -> Tuple[float, ...]:
    """Encode exactly the signals `operation_classifier.classify_feature()`
    itself reads: primitive type (one-hot), log-scaled radius, whether a
    principal axis exists, and the two axis-relationship booleans the rule
    engine already computes (`is_coaxial`, `is_axis_aligned_with_any`)."""
    ptype = primitive.type
    is_cylindrical = 1.0 if ptype == "cylindrical" else 0.0
    is_conical = 1.0 if ptype == "conical" else 0.0
    is_planar = 1.0 if ptype == "planar" else 0.0
    is_unknown_type = 1.0 if ptype not in ("cylindrical", "conical", "planar") else 0.0

    radius = primitive.details.get("radius") or 0.0
    log_radius = math.log1p(radius)

    has_principal_axis = 1.0 if principal_axis is not None else 0.0

    primitive_axis = axis_from_details(primitive.details)
    is_coaxial_with_principal = 0.0
    is_axis_aligned = 0.0
    if primitive_axis is not None:
        if principal_axis is not None:
            is_coaxial_with_principal = 1.0 if is_coaxial(primitive_axis, principal_axis) else 0.0
        is_axis_aligned = 1.0 if is_axis_aligned_with_any(primitive_axis.direction) else 0.0

    return (
        is_cylindrical, is_conical, is_planar, is_unknown_type,
        log_radius, has_principal_axis, is_coaxial_with_principal, is_axis_aligned,
    )


def _passthrough_features(primitives: List[SurfacePrimitive]) -> List[Feature]:
    """Each `.scad`/`.py`-ingested primitive is its own feature, per the
    established `add-cnc-operation-classifier` pattern (see
    scripts/classify_directory.py) — there is no further face-grouping
    ambiguity to resolve at this ingestion granularity."""
    return [Feature(feature_type=p.type, face_ids=[p.face_id], parameters={}) for p in primitives]


def build_step_examples(step_paths: List[Union[str, Path]]) -> List[TrainingExample]:
    """Label examples using StepAnalyzer.analyze()'s own report — its
    per-feature `operation` field is already `classify_feature()`'s output
    for that exact primitive with `principal_axis=None` (the STEP path
    never computes one), so there is no need to recompute it separately."""
    from .pipeline import StepAnalyzer  # local import: avoids a circular

    examples: List[TrainingExample] = []
    for path in step_paths:
        path = str(path)
        try:
            report = StepAnalyzer().analyze(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        primitives_by_face = {p["face_id"]: p for p in report["primitives"]}
        for feature in report["features"]:
            if "operation" not in feature:
                continue
            prim_dicts = [primitives_by_face[fid] for fid in feature["face_ids"] if fid in primitives_by_face]
            if not prim_dicts:
                continue
            prim = SurfacePrimitive(
                face_id=prim_dicts[0]["face_id"],
                type=prim_dicts[0]["type"],
                details=prim_dicts[0]["details"],
            )
            vector = vectorize(prim, principal_axis=None)
            examples.append(TrainingExample(vector=vector, label=feature["operation"], source=Path(path).name))

    logger.info("STEP corpus: %d examples from %d file(s)", len(examples), len(step_paths))
    return examples


def build_scad_examples(library_root: Union[str, Path]) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    entries = scad_ingest.ingest_directory(library_root)
    for entry in entries:
        if entry.error is not None or entry.result is None:
            continue
        result = entry.result
        features = _passthrough_features(result.primitives)
        feature_ops = classify_features(features, result.primitives, result.principal_axis)
        primitives_by_face = {p.face_id: p for p in result.primitives}
        for fo in feature_ops:
            prim = primitives_by_face.get(fo.feature.face_ids[0])
            if prim is None:
                continue
            vector = vectorize(prim, result.principal_axis)
            examples.append(TrainingExample(vector=vector, label=fo.operation, source=entry.path.name))

    logger.info("OpenSCAD corpus: %d examples from %d discovered file(s)", len(examples), len(entries))
    return examples


def build_freecad_examples(library_root: Union[str, Path]) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    entries = py_source_ingest.ingest_directory(library_root)
    for entry in entries:
        if entry.error is not None or entry.result is None:
            continue
        result = entry.result
        features = _passthrough_features(result.primitives)
        feature_ops = classify_features(features, result.primitives, result.principal_axis)
        primitives_by_face = {p.face_id: p for p in result.primitives}
        for fo in feature_ops:
            prim = primitives_by_face.get(fo.feature.face_ids[0])
            if prim is None:
                continue
            vector = vectorize(prim, result.principal_axis)
            examples.append(TrainingExample(vector=vector, label=fo.operation, source=entry.path.name))

    logger.info("FreeCAD corpus: %d examples from %d discovered file(s)", len(examples), len(entries))
    return examples


def build_corpus(
    nist_paths: List[Union[str, Path]],
    scad_root: Union[str, Path],
    freecad_root: Union[str, Path],
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    examples.extend(build_step_examples(nist_paths))
    examples.extend(build_scad_examples(scad_root))
    examples.extend(build_freecad_examples(freecad_root))

    counts: dict = {}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1
    logger.info("Corpus assembled: %d examples total. Per-class counts: %s", len(examples), counts)

    return examples

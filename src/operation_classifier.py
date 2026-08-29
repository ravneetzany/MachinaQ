"""Classify detected features (and a part as a whole) into required CNC operation types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .features import Feature
from .geometry import Axis, axis_from_details, is_axis_aligned_with_any, is_coaxial
from .primitive import SurfacePrimitive


class Operation:
    """Required CNC operation type constants."""

    TURNING = "turning"
    DRILLING = "drilling"
    FACE_MILLING = "face_milling"
    THREE_AXIS_MILLING = "3_axis_milling"
    FIVE_AXIS_MILLING = "5_axis_milling"
    UNKNOWN = "unknown"

    ALL = (TURNING, DRILLING, FACE_MILLING, THREE_AXIS_MILLING, FIVE_AXIS_MILLING, UNKNOWN)


@dataclass
class FeatureOperation:
    feature: Feature
    operation: str
    rationale: str


@dataclass
class PartOperationsSummary:
    primary_process: str
    secondary_processes: List[str]
    rationale: str
    feature_operations: List[FeatureOperation]


def _primitive_by_face(primitives: List[SurfacePrimitive]) -> Dict[int, SurfacePrimitive]:
    return {p.face_id: p for p in primitives}


def _axes_of(prims: List[SurfacePrimitive]) -> List[Axis]:
    axes = [axis_from_details(p.details) for p in prims]
    return [a for a in axes if a is not None]


def classify_feature(
    feature: Feature,
    primitives_by_face: Dict[int, SurfacePrimitive],
    principal_axis: Optional[Axis],
) -> FeatureOperation:
    """Assign one required CNC operation to a single feature.

    When `principal_axis` is None (the STEP-derived path, which does not yet
    compute a part-wide axis), falls back to coarser feature-type-only
    heuristics per design.md decision 2.
    """
    prims = [primitives_by_face[fid] for fid in feature.face_ids if fid in primitives_by_face]
    if not prims:
        return FeatureOperation(
            feature, Operation.UNKNOWN,
            "no matching geometric primitive found for this feature's face_ids; insufficient geometric data",
        )

    ptypes = {p.type for p in prims}
    if "unknown" in ptypes:
        return FeatureOperation(
            feature, Operation.UNKNOWN,
            "underlying primitive type is unknown; insufficient geometric data",
        )

    if principal_axis is None:
        return _classify_without_axis(feature, prims, ptypes)
    return _classify_with_axis(feature, prims, ptypes, principal_axis)


def _classify_without_axis(feature: Feature, prims: List[SurfacePrimitive], ptypes: set) -> FeatureOperation:
    """No part-wide principal axis is available — either because the part
    genuinely has none (a prismatic part, per the spec's "Cylindrical or
    conical hole not coaxial with the principal axis" and "Planar feature
    on a prismatic part" scenarios), or because the STEP path doesn't yet
    compute one at all (design.md decision 2's documented lower-confidence
    fallback). Both cases use the same primitive-type-only heuristic: a
    cylindrical/conical face with no axis to be coaxial with cannot be a
    turning feature, so it reads as drilling; a planar face reads as
    3-axis milling."""
    if "cylindrical" in ptypes or "conical" in ptypes:
        return FeatureOperation(
            feature, Operation.DRILLING,
            "no part-wide principal rotational axis available, so this "
            f"{'cylindrical' if 'cylindrical' in ptypes else 'conical'} feature cannot be "
            "coaxial with one; classified as drilling from primitive type alone",
        )
    if "planar" in ptypes:
        return FeatureOperation(
            feature, Operation.THREE_AXIS_MILLING,
            "planar primitive on a part with no single principal rotational axis; "
            "classified as 3-axis milling from primitive type alone",
        )
    if "toroidal" in ptypes:
        toroidal_axes = _axes_of([p for p in prims if p.type == "toroidal"])
        if toroidal_axes and all(is_axis_aligned_with_any(a.direction) for a in toroidal_axes):
            return FeatureOperation(
                feature, Operation.THREE_AXIS_MILLING,
                "toroidal face axis is aligned with an orthogonal machine axis; "
                "reachable via 3-axis milling",
            )
        return FeatureOperation(
            feature, Operation.FIVE_AXIS_MILLING,
            "toroidal face axis is not aligned with an orthogonal machine axis (or "
            "unresolvable); requires non-orthogonal tool access",
        )
    if "freeform" in ptypes:
        return FeatureOperation(
            feature, Operation.FIVE_AXIS_MILLING,
            "free-form (B-spline) primitive carries no resolvable axis/normal data by "
            "design; conservatively classified as requiring 5-axis milling",
        )
    return FeatureOperation(
        feature, Operation.UNKNOWN,
        "primitive type does not map to a known operation heuristic",
    )


def _classify_with_axis(
    feature: Feature,
    prims: List[SurfacePrimitive],
    ptypes: set,
    principal_axis: Axis,
) -> FeatureOperation:
    if "cylindrical" in ptypes:
        cyl_prims = [p for p in prims if p.type == "cylindrical"]
        axes = _axes_of(cyl_prims)
        if axes and all(is_coaxial(a, principal_axis) for a in axes):
            return FeatureOperation(
                feature, Operation.TURNING,
                "cylindrical face coaxial with the part's principal rotational axis",
            )
        if axes and all(is_axis_aligned_with_any(a.direction) for a in axes):
            return FeatureOperation(
                feature, Operation.DRILLING,
                "cylindrical hole axis is not coaxial with the part's principal axis, "
                "but is aligned with an orthogonal machine axis",
            )
        if axes:
            return FeatureOperation(
                feature, Operation.FIVE_AXIS_MILLING,
                "cylindrical feature axis is neither coaxial with the part's principal axis "
                "nor aligned with an orthogonal machine axis; requires non-orthogonal tool access",
            )
        return FeatureOperation(
            feature, Operation.DRILLING,
            "cylindrical hole-type feature with no part-wide-coaxial axis and no per-primitive "
            "axis data available",
        )

    if "conical" in ptypes:
        return FeatureOperation(
            feature, Operation.DRILLING,
            "conical primitive is typically produced by a drilling/countersinking operation",
        )

    if "planar" in ptypes:
        planar_prims = [p for p in prims if p.type == "planar"]
        axes = _axes_of(planar_prims)
        if axes and all(is_axis_aligned_with_any(a.direction) for a in axes):
            return FeatureOperation(
                feature, Operation.THREE_AXIS_MILLING,
                "planar face normal is aligned with an orthogonal machine axis; "
                "reachable via face/3-axis milling",
            )
        if axes:
            return FeatureOperation(
                feature, Operation.FIVE_AXIS_MILLING,
                "planar face normal is neither coaxial with the part's principal axis nor "
                "aligned with an orthogonal machine axis; requires non-orthogonal tool access",
            )
        return FeatureOperation(
            feature, Operation.FACE_MILLING,
            "planar feature on a part with no single principal rotational axis",
        )

    return FeatureOperation(
        feature, Operation.UNKNOWN,
        "primitive type does not map to a known operation heuristic",
    )


def classify_features(
    features: List[Feature],
    primitives: List[SurfacePrimitive],
    principal_axis: Optional[Axis],
) -> List[FeatureOperation]:
    by_face = _primitive_by_face(primitives)
    return [classify_feature(f, by_face, principal_axis) for f in features]


def summarize_part(feature_operations: List[FeatureOperation]) -> PartOperationsSummary:
    """Roll per-feature operations up into one part-level primary/secondary summary.

    `unknown` features are excluded from the vote. When at least one feature is
    classified `turning`, it is treated as the primary process regardless of
    feature counts — a turned base body typically has one dominant cylindrical
    surface but may carry several small secondary (e.g. milled) features, and
    counting features alone would misidentify the base process.
    """
    counted = [fo for fo in feature_operations if fo.operation != Operation.UNKNOWN]

    if not counted:
        return PartOperationsSummary(
            primary_process=Operation.UNKNOWN,
            secondary_processes=[],
            rationale="no feature could be classified with sufficient geometric evidence",
            feature_operations=feature_operations,
        )

    counts: Dict[str, int] = {}
    for fo in counted:
        counts[fo.operation] = counts.get(fo.operation, 0) + 1

    if Operation.TURNING in counts:
        primary_op = Operation.TURNING
    else:
        primary_op = max(counts.items(), key=lambda item: item[1])[0]

    secondary_ops = [op for op in counts if op != primary_op]

    if not secondary_ops:
        rationale = f"all classified features require {primary_op}"
    elif primary_op == Operation.TURNING:
        rationale = (
            f"part's base body is axisymmetric and classified {primary_op}; "
            f"{', '.join(secondary_ops)} required for feature(s) that cannot be produced by "
            f"{primary_op} alone"
        )
    else:
        rationale = (
            f"{primary_op} required by the largest share of features "
            f"({counts[primary_op]}/{len(counted)}); "
            f"{', '.join(secondary_ops)} required for the remaining feature(s)"
        )

    return PartOperationsSummary(
        primary_process=primary_op,
        secondary_processes=secondary_ops,
        rationale=rationale,
        feature_operations=feature_operations,
    )

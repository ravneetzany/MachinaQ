"""Rule-based manufacturing feature detection.

`hole`/`thread` features are sourced elsewhere (`StepAnalyzer._hole_and_thread_features`,
from the parser's standards-validated hole detection) — this module classifies
the remaining primitives (boss/slot/drill/planar_face) using more than primitive type
alone, so every face gets at most one feature label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .primitive import SurfacePrimitive

logger = logging.getLogger(__name__)

#: Long/short bounding-extent ratio at or above which a planar face is
#: considered a narrow, elongated pocket rather than a general flat face.
SLOT_ASPECT_RATIO_THRESHOLD = 3.0
#: A slot's bottom face is bounded by a handful of wall faces (e.g. 4 for a
#: rectangular pocket) — too few suggests an open edge, too many suggests a
#: large exterior face rather than a pocket.
SLOT_MIN_ADJACENT_FACES = 2
SLOT_MAX_ADJACENT_FACES = 6
#: Relative radius difference under which a conical face's adjacent
#: cylindrical face counts as the same countersink/pilot-hole feature.
DRILL_RADIUS_TOLERANCE = 0.2

#: Minimum extents (same units as `long_extent`/`short_extent`, mm for
#: STEP-derived parts) a planar face's bounding extents must independently
#: exceed to be classified as a large `face` feature. Two independent
#: minimums, not an area threshold, so a narrow-but-long slot (small
#: `short_extent`) can't qualify via `long_extent` alone — see design.md
#: decision 2 in the add-face-milling-feature-detection change.
FACE_MIN_LONG_EXTENT = 10.0
FACE_MIN_SHORT_EXTENT = 10.0


@dataclass
class Feature:
    feature_type: str
    face_ids: List[int]
    parameters: Dict[str, Any]


class FeatureDetector:
    def __init__(self) -> None:
        pass

    def detect_bosses(
        self, primitives: List[SurfacePrimitive], hole_face_ids: Set[int]
    ) -> List[Feature]:
        """A cylindrical/conical face with exactly one adjacent planar face,
        and not already a standards-matched hole. Cannot yet distinguish a
        true convex protrusion from a concave pocket sharing the same
        topology signature — see design.md's documented limitation."""
        bosses: List[Feature] = []
        for p in primitives:
            if p.type not in {"cylindrical", "conical"}:
                continue
            if p.face_id in hole_face_ids:
                continue
            if p.details.get("adjacent_planar_count", 0) == 1:
                bosses.append(Feature(
                    feature_type="boss",
                    face_ids=[p.face_id],
                    parameters={
                        **p.details,
                        "rationale": (
                            f"{p.type} face has exactly one adjacent planar face and is "
                            "not a standards-matched hole (protrusion evidence)"
                        ),
                    },
                ))
        return bosses

    def detect_slots(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        slots: List[Feature] = []
        for p in primitives:
            if p.type != "planar":
                continue
            long_extent = p.details.get("long_extent")
            short_extent = p.details.get("short_extent")
            if not long_extent or not short_extent:
                continue
            aspect_ratio = long_extent / short_extent
            adjacent_count = len(p.adjacent_face_ids)
            if not (SLOT_MIN_ADJACENT_FACES <= adjacent_count <= SLOT_MAX_ADJACENT_FACES):
                continue
            if aspect_ratio < SLOT_ASPECT_RATIO_THRESHOLD:
                continue
            slots.append(Feature(
                feature_type="slot",
                face_ids=[p.face_id],
                parameters={
                    **p.details,
                    "aspect_ratio": aspect_ratio,
                    "rationale": (
                        f"planar face bounded by {adjacent_count} adjacent faces, "
                        f"long/short extent ratio {aspect_ratio:.1f} at or above the "
                        f"narrow-pocket threshold ({SLOT_ASPECT_RATIO_THRESHOLD})"
                    ),
                },
            ))
        return slots

    def detect_drills(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        drills: List[Feature] = []
        for p in primitives:
            if p.type != "conical":
                continue
            radius = p.details.get("radius")
            nearest = p.details.get("nearest_adjacent_cylindrical_radius")
            if radius is None or nearest is None:
                continue
            denom = max(radius, nearest, 1e-9)
            if abs(radius - nearest) / denom > DRILL_RADIUS_TOLERANCE:
                continue
            drills.append(Feature(
                feature_type="drill",
                face_ids=[p.face_id],
                parameters={
                    **p.details,
                    "rationale": (
                        "conical face adjacent to a cylindrical face of compatible radius "
                        f"({nearest:.3f} vs {radius:.3f}); countersink/pilot-hole pattern"
                    ),
                },
            ))
        return drills

    def detect_planar_faces(self, primitives: List[SurfacePrimitive]) -> List[Feature]:
        """A planar face not already claimed as a slot (or anything else),
        whose bounding extents both independently clear the large-face
        thresholds — e.g. a part's top stock face. Named `planar_face`, not
        `face`/`face_milling`: this is a *geometric* feature classification,
        distinct from `operation_classifier.Operation.FACE_MILLING` (a
        required-*operation* label decided separately, downstream, by
        `classify_feature`) — the two must not share a name. See design.md
        decision 2 in the add-face-milling-feature-detection change for why
        this uses two independent minimums rather than an area or
        aspect-ratio check."""
        planar_faces: List[Feature] = []
        for p in primitives:
            if p.type != "planar":
                continue
            long_extent = p.details.get("long_extent")
            short_extent = p.details.get("short_extent")
            if not long_extent or not short_extent:
                continue
            if long_extent < FACE_MIN_LONG_EXTENT or short_extent < FACE_MIN_SHORT_EXTENT:
                continue
            planar_faces.append(Feature(
                feature_type="planar_face",
                face_ids=[p.face_id],
                parameters={
                    **p.details,
                    "rationale": (
                        f"planar face with extents {long_extent:.1f} x {short_extent:.1f} both at "
                        f"or above the large-face thresholds ({FACE_MIN_LONG_EXTENT:.1f} x "
                        f"{FACE_MIN_SHORT_EXTENT:.1f}); not claimed by any narrower-evidence rule"
                    ),
                },
            ))
        return planar_faces

    def detect_all_features(
        self,
        primitives: List[SurfacePrimitive],
        claimed_face_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[Feature], List[int]]:
        """Classify primitives not already claimed elsewhere (holes/threads)
        into boss/slot/drill features, each face getting at most one label.
        Returns (features, unclassified_face_ids).
        """
        claimed_face_ids = claimed_face_ids or set()
        candidates = [p for p in primitives if p.face_id not in claimed_face_ids]

        drills = self.detect_drills(candidates)
        drill_face_ids = {f.face_ids[0] for f in drills}

        remaining = [p for p in candidates if p.face_id not in drill_face_ids]
        bosses = self.detect_bosses(remaining, claimed_face_ids)
        boss_face_ids = {f.face_ids[0] for f in bosses}

        remaining = [p for p in remaining if p.face_id not in boss_face_ids]
        slots = self.detect_slots(remaining)
        slot_face_ids = {f.face_ids[0] for f in slots}

        remaining = [p for p in remaining if p.face_id not in slot_face_ids]
        planar_faces = self.detect_planar_faces(remaining)
        planar_face_ids = {f.face_ids[0] for f in planar_faces}

        matched_face_ids = drill_face_ids | boss_face_ids | slot_face_ids | planar_face_ids
        unclassified_face_ids = [p.face_id for p in candidates if p.face_id not in matched_face_ids]

        features = drills + bosses + slots + planar_faces
        logger.debug(
            "Detected %d features (%d drills, %d bosses, %d slots, %d planar_faces), %d unclassified",
            len(features), len(drills), len(bosses), len(slots), len(planar_faces), len(unclassified_face_ids),
        )
        return features, unclassified_face_ids

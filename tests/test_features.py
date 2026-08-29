from src.features import DRILL_RADIUS_TOLERANCE, SLOT_ASPECT_RATIO_THRESHOLD, FeatureDetector
from src.primitive import SurfacePrimitive


def test_planar_face_with_no_slot_evidence_is_unclassified() -> None:
    # Large, low-aspect-ratio face (e.g. a top face) — should not be a slot.
    prim = SurfacePrimitive(
        face_id=1, type="planar",
        details={"long_extent": 20.0, "short_extent": 18.0},
        adjacent_face_ids=[2, 3, 4],
    )
    features, unclassified = FeatureDetector().detect_all_features([prim])
    assert features == []
    assert unclassified == [1]


def test_planar_face_with_qualifying_slot_evidence_is_a_slot() -> None:
    prim = SurfacePrimitive(
        face_id=1, type="planar",
        details={"long_extent": 30.0, "short_extent": 5.0},  # ratio 6 >= threshold
        adjacent_face_ids=[2, 3, 4, 5],
    )
    features, unclassified = FeatureDetector().detect_all_features([prim])
    assert len(features) == 1
    assert features[0].feature_type == "slot"
    assert features[0].face_ids == [1]
    assert unclassified == []


def test_face_receives_exactly_one_label() -> None:
    prim = SurfacePrimitive(
        face_id=1, type="planar",
        details={"long_extent": 30.0, "short_extent": 5.0},
        adjacent_face_ids=[2, 3],
    )
    features, _ = FeatureDetector().detect_all_features([prim])
    matches = [f for f in features if 1 in f.face_ids]
    assert len(matches) == 1


def test_boss_requires_one_adjacent_planar_face_and_not_a_hole() -> None:
    boss_prim = SurfacePrimitive(
        face_id=10, type="cylindrical",
        details={"radius": 5.0, "adjacent_planar_count": 1.0},
    )
    detector = FeatureDetector()
    features, unclassified = detector.detect_all_features([boss_prim], claimed_face_ids=set())
    assert len(features) == 1
    assert features[0].feature_type == "boss"


def test_boss_excludes_faces_already_claimed_as_holes() -> None:
    prim = SurfacePrimitive(
        face_id=10, type="cylindrical",
        details={"radius": 5.0, "adjacent_planar_count": 1.0},
    )
    detector = FeatureDetector()
    features, unclassified = detector.detect_all_features([prim], claimed_face_ids={10})
    assert features == []
    assert unclassified == []  # excluded entirely, not "unclassified" either


def test_cylindrical_face_without_boss_evidence_is_unclassified() -> None:
    prim = SurfacePrimitive(
        face_id=10, type="cylindrical",
        details={"radius": 5.0, "adjacent_planar_count": 0.0},
    )
    features, unclassified = FeatureDetector().detect_all_features([prim])
    assert features == []
    assert unclassified == [10]


def test_drill_requires_adjacent_compatible_radius_cylindrical_face() -> None:
    prim = SurfacePrimitive(
        face_id=20, type="conical",
        details={"radius": 4.0, "semi_angle": 0.5, "nearest_adjacent_cylindrical_radius": 4.1},
    )
    features, _ = FeatureDetector().detect_all_features([prim])
    assert len(features) == 1
    assert features[0].feature_type == "drill"


def test_standalone_conical_face_without_pilot_hole_is_unclassified() -> None:
    prim = SurfacePrimitive(
        face_id=20, type="conical",
        details={"radius": 4.0, "semi_angle": 0.5},
    )
    features, unclassified = FeatureDetector().detect_all_features([prim])
    assert features == []
    assert unclassified == [20]


def test_every_feature_has_a_rationale() -> None:
    primitives = [
        SurfacePrimitive(face_id=1, type="planar",
                          details={"long_extent": 30.0, "short_extent": 5.0},
                          adjacent_face_ids=[2, 3]),
        SurfacePrimitive(face_id=10, type="cylindrical",
                          details={"radius": 5.0, "adjacent_planar_count": 1.0}),
        SurfacePrimitive(face_id=20, type="conical",
                          details={"radius": 4.0, "semi_angle": 0.5,
                                   "nearest_adjacent_cylindrical_radius": 4.1}),
    ]
    features, _ = FeatureDetector().detect_all_features(primitives)
    assert len(features) == 3
    for feature in features:
        assert feature.parameters.get("rationale")

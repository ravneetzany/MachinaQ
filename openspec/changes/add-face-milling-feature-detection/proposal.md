## Why

`FeatureDetector.detect_all_features()` (`src/features.py`) only recognizes `boss`/`slot`/`drill` features from primitives not already claimed by hole/thread detection. A large flat planar face (e.g. a part's top stock face) is neither a hole/thread, nor a boss (not cylindrical/conical), nor a slot (not narrow/elongated) — so it falls into `unclassified_face_ids` and never becomes a `Feature` at all. `operation_classifier.classify_feature()` already has correct logic to classify a planar primitive's required operation (`3_axis_milling` if its normal is axis-aligned, `5_axis_milling` otherwise, `face_milling` in the no-principal-axis edge case) — but that logic never runs for these faces, since no `Feature` ever references them. Real-world use of the FreeCAD CAM addon surfaced this directly: selecting a large flat face on a test part showed no recognized feature or operation for it at all in the classification report, only unrelated hole/thread entries for nearby features.

## What Changes

- Add a new `face` feature type to `FeatureDetector`, detecting large planar faces not already claimed by hole/thread/boss/slot detection, using a size/extent heuristic (reusing `SurfacePrimitive.details`' existing extent fields) to distinguish a genuine large flat face from smaller/ambiguous planar surfaces.
- Wire `face` detection into `detect_all_features()`'s existing claim order (after drills/bosses/slots, before falling back to unclassified) and into `pipeline.py` wherever `detect_all_features()`'s output is consumed, so these features reach `operation_classifier.classify_feature()` unchanged.
- No changes to `operation_classifier.py`'s classification logic — its existing planar-primitive handling already produces the correct operation once a `Feature` exists to feed it.

## Capabilities

### Modified Capabilities
- `manufacturing-feature-detection`: adds a new `face` feature type for large planar faces not otherwise claimed, and narrows the existing "unclassified" guarantee to still apply only to faces with no qualifying evidence for any rule (including this new one).

## Impact

- `src/features.py`: new `detect_faces()` method (or equivalent) and updated `detect_all_features()` claim order.
- `src/pipeline.py`: no interface change expected (features already flow through the same list), but the classify-then-summarize call sites need re-verifying since previously-unclassified large planar faces will now appear as real features/operations in `PartOperationsSummary`.
- `tests/`: new unit tests for the `face` detection rule; existing tests asserting a large planar face is `unclassified` may need updating if any rely on that as a fixture (to be checked during design/implementation, not assumed).
- `freecad_addon/MachinaQCAM/`: no code changes required — once the core classifier reports these features, the existing correlation/panel code surfaces them automatically.

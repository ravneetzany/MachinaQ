## 1. Parser: multi-line entity fix and new primitive extraction

- [x] 1.1 In `src/parser.py`'s `_parse_entities()`, add `re.DOTALL` to the entity-matching regex; verify with a unit test parsing a synthetic multi-line STEP entity (attributes spanning embedded newlines before the terminating `;`) and asserting it appears in `self.entities` with the correct id/type
  - Added `test_multiline_entity_is_parsed_not_dropped` to `tests/test_parser_geometry.py`; passing.
- [x] 1.2 Run the full existing test suite with only the 1.1 fix applied (no new primitive types yet) and review any count/behavior changes in existing STEP-fixture tests before proceeding, per design.md's Risks section; note findings in this task's completion comment
  - `.venv/bin/python -m pytest tests/ --ignore=tests/test_parser.py -q` — 93/93 passing (up from 92, the new test). No existing test's assertions changed behavior — none of the current STEP fixtures apparently rely on multi-line entities in ways that shift primitive/feature counts. Safe to proceed.
- [x] 1.3 Add `GeometryPrimitives.toroids: List[Tuple[int, float, float, Vector3, Vector3]]` (surface id, major radius, minor radius, axis point, axis direction) and a `TOROIDAL_SURFACE` branch in `_extract_primitives()` resolving axis placement via the existing `_resolve_placement()`, mirroring `CONICAL_SURFACE`'s structure (design.md decision 2); verify with a unit test parsing a STEP file/synthetic entity containing a `TOROIDAL_SURFACE` and asserting a resolved `(major_radius, minor_radius, point, direction)` tuple is present
  - Added `test_toroidal_surface_extracted_with_resolved_axis` to `tests/test_parser_geometry.py`; passing.
- [x] 1.4 Add `GeometryPrimitives.freeforms: List[int]` (surface ids only) and a `B_SPLINE_SURFACE_WITH_KNOTS` branch in `_extract_primitives()` recording the surface id without parsing its nested attributes (design.md decision 3); verify with a unit test asserting a synthetic multi-line `B_SPLINE_SURFACE_WITH_KNOTS` entity's surface id appears in `parser.primitives.freeforms`
  - Added `test_bspline_surface_recorded_as_freeform_surface_id` to `tests/test_parser_geometry.py`; passing.

## 2. Pipeline: build toroidal/freeform SurfacePrimitives

- [x] 2.1 In `src/pipeline.py`'s `_extract_primitives()`, add a branch building `SurfacePrimitive(type="toroidal", details={"major_radius": ..., "minor_radius": ..., **axis_to_details(...)}, ...)` for each entry in `parser.primitives.toroids`, mirroring the existing cylinder/cone loops; verify with a unit test asserting a `toroidal`-typed `SurfacePrimitive` round-trips through `axis_from_details()` to a valid `Axis`
  - Resolved design.md's Open Question: also gave `toroidal` primitives `long_extent`/`short_extent` (via `get_face_bounding_extents(fid, normal=None)`, same as `freeform`), since it's cheap and gives `elongated_boss` detection real evidence for this type too. Added `test_toroidal_primitive_round_trips_axis` to `tests/test_pipeline.py`; passing.
- [x] 2.2 In the same method, add a branch building `SurfacePrimitive(type="freeform", details={"long_extent": ..., "short_extent": ...} or {} if unresolvable, ...)` for each entry in `parser.primitives.freeforms`, calling `get_face_bounding_extents(face_id, normal=None)` per `get_surface_face_ids(surf_id)` face (design.md decision 3, mirroring the existing plane loop's structure); verify with a unit test asserting a `freeform`-typed `SurfacePrimitive` carries `long_extent`/`short_extent` when the underlying face has resolvable vertices, and an empty `details` dict (not an error) when it doesn't
  - Added `test_freeform_primitive_carries_extents_when_resolvable` and `test_freeform_primitive_empty_details_when_unresolvable` to `tests/test_pipeline.py`; passing.

## 3. Feature detection: elongated_boss

- [x] 3.1 In `src/features.py`, resolve design.md's Open Question (extent data source for `cylindrical`/`toroidal` primitives in this rule) using whichever real fixture data is available, and add `FeatureDetector.detect_elongated_bosses()`: classify `toroidal`/`freeform`/`cylindrical` primitives not already claimed by hole/thread/boss/slot/planar_face detection as `elongated_boss` when their extent aspect ratio meets an elongation threshold (a new module-level constant, e.g. `ELONGATED_BOSS_ASPECT_RATIO_THRESHOLD`), with a rationale string; primitives with no resolvable extent data are left unclassified by this rule (design.md's Risks section), not errored
  - Open Question resolved in task 2.1: `toroidal` primitives now get `long_extent`/`short_extent` too. `cylindrical` primitives still don't (unchanged, out of this change's scope), so the rule naturally leaves them unclassified via the missing-extent guard rather than needing special-case logic.
- [x] 3.2 Wire `detect_elongated_bosses()` into `detect_all_features()`'s claim order, last (after drills/bosses/slots/planar_face, per design.md decision 4); update the debug log line's counts
- [x] 3.3 Verify with unit tests in `tests/test_features.py`: a `toroidal`/`freeform`/`cylindrical` primitive with elongated extents and not otherwise claimed is classified `elongated_boss`; a primitive already claimed by an earlier rule (hole/boss/slot/planar_face) is not double-claimed as `elongated_boss`; a primitive with no resolvable extent data remains unclassified rather than erroring; a primitive with extents below the elongation threshold remains unclassified
  - Added 6 new tests; 19/19 passing in `tests/test_features.py`.

## 4. Operation classification: toroidal and freeform

- [x] 4.1 In `src/operation_classifier.py`'s `_classify_without_axis()`, add a `toroidal` branch reusing `is_axis_aligned_with_any()` on the primitive's resolved axis direction (design.md decision 5): axis-aligned → `THREE_AXIS_MILLING`, otherwise → `FIVE_AXIS_MILLING`
- [x] 4.2 Add a `freeform` branch unconditionally returning `FIVE_AXIS_MILLING` (design.md decision 5), with a rationale noting the absence of resolvable axis/normal data
- [x] 4.3 Verify with unit tests in `tests/test_operation_classifier.py`: an axis-aligned `toroidal` primitive classifies `3_axis_milling`; a non-axis-aligned `toroidal` primitive classifies `5_axis_milling`; a `freeform` primitive classifies `5_axis_milling` regardless of any other data present
  - Added 3 new tests; 13/13 passing in `tests/test_operation_classifier.py`.

## 5. End-to-end verification

- [x] 5.1 Run the full test suite and confirm no unexpected regressions: `.venv/bin/python -m pytest tests/ --ignore=tests/test_parser.py -q`
  - 107/107 passing (up from 93 before this change).
- [x] 5.2 Verify end-to-end against the real STEP export that surfaced this change (or an equivalent fixture with `B_SPLINE_SURFACE_WITH_KNOTS`/`TOROIDAL_SURFACE` faces): the previously-invisible curved arm face now appears in the report as an `elongated_boss` feature with a non-`unknown` operation, not matched via the correlation panel's distant-primitive fallback
  - Re-ran `StepAnalyzer().analyze()` on `/tmp/machinaq_l0k43u8l/export.step` (the exact export from the user's live session): 16 `elongated_boss` features detected, including face_ids 1052/1195 (the same `B_SPLINE_SURFACE_WITH_KNOTS` entities identified during the propose phase), each with `operation: "5_axis_milling"`. Part-level `operations_summary` now reports `3_axis_milling` as primary with `drilling`/`5_axis_milling` as secondary, instead of `drilling`-only.

## 6. Docs

- [x] 6.1 Update `README.md`'s feature-type list and `src/features.py`'s file-tree comment (same locations touched by the prior `planar_face` change) to include `elongated_boss`; verify by re-reading the updated sections

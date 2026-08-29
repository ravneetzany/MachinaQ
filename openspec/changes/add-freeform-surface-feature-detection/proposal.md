## Why

Real-world FreeCAD CAM addon testing surfaced a parser gap, plus an actual parser bug underneath it: `src/parser.py`'s `_extract_primitives()` only recognizes `PLANE`, `CYLINDRICAL_SURFACE`, and `CONICAL_SURFACE` STEP entities — a test part with curved, elongated cross-shaped arms (built via a swept/lofted profile) exports a STEP file containing `B_SPLINE_SURFACE_WITH_KNOTS` (88 instances) and `TOROIDAL_SURFACE` (36 instances) entities that are never extracted as primitives at all, so faces made of them never reach feature detection and never even appear in `unclassified_face_ids` — they vanish before any of that runs. Investigating why turned up a deeper bug: `_parse_entities()`'s entity-matching regex uses `.` without `re.DOTALL`, so any STEP entity whose definition spans multiple lines (as `B_SPLINE_SURFACE_WITH_KNOTS`'s control-point/knot-vector attributes routinely do) silently fails to match and is dropped from `self.entities` entirely — confirmed directly: 0/88 `B_SPLINE_SURFACE_WITH_KNOTS` entities parse today, vs. 88/88 once `re.DOTALL` is added. `TOROIDAL_SURFACE` (single-line) already parses fine at the entity level (36/36); it's only missing from `_extract_primitives()`'s dispatch.

Selecting a face made of either surface type in the FreeCAD addon's face-correlation panel currently falls back to matching a distant, unrelated primitive (e.g. a small conical hole primitive tens of mm away), which is misleading. The user's expectation for the real part that surfaced this: the curved elongated arm should be recognized as a boss-like feature (an "elongated"/"extended" boss — a protrusion, not a cavity) requiring a milling operation (3-axis or 5-axis).

## What Changes

- Fix `_parse_entities()`'s multi-line entity-matching bug (add `re.DOTALL` or equivalent) so entities like `B_SPLINE_SURFACE_WITH_KNOTS` are captured at all — a correctness fix independent of the rest of this change, since any multi-line STEP entity of any type was previously silently dropped.
- Add primitive extraction for `TOROIDAL_SURFACE` (axis + major/minor radius, via the same `_resolve_placement` helper `PLANE`/`CYLINDRICAL_SURFACE`/`CONICAL_SURFACE` already use) and `B_SPLINE_SURFACE_WITH_KNOTS` (represented only by its face's bounding extents via the existing, surface-agnostic `get_face_bounding_extents()` — no NURBS math attempted, matching how `long_extent`/`short_extent` already work for planar faces).
- Add an `elongated_boss` feature-detection rule to `FeatureDetector`, using bounding-extent aspect ratio (the same evidence shape `detect_slots()` already uses, but for a protrusion rather than a cavity) over the new `toroidal`/`freeform` primitive types plus the existing `cylindrical` type.
- Extend `operation_classifier.py`'s primitive-type-only (`_classify_without_axis`) path to give `toroidal`/`freeform` primitives a required-operation determination — axis-alignment-based (3-axis vs. 5-axis) for `toroidal` (which retains real axis data), defaulting to 5-axis for `freeform` (which, by design, carries no axis data) — instead of leaving them `unknown`.

## Capabilities

### Modified Capabilities
- `manufacturing-feature-detection`: adds `elongated_boss` as a new feature type, sourced from new `toroidal`/`freeform` primitive types.

## Impact

- `src/parser.py`: entity-parsing regex fix (affects all multi-line STEP entities, not just the two types this change targets); new `GeometryPrimitives.toroids`/`.freeforms` lists; new `_extract_primitives()` branches for `TOROIDAL_SURFACE`/`B_SPLINE_SURFACE_WITH_KNOTS`.
- `src/pipeline.py`: `_extract_primitives()` gains branches building `SurfacePrimitive(type="toroidal", ...)`/`SurfacePrimitive(type="freeform", ...)` entries, mirroring the existing cylinder/cone/plane branches.
- `src/features.py`: new `detect_elongated_bosses()` (or equivalent) method wired into `detect_all_features()`'s claim order.
- `src/operation_classifier.py`: `_classify_without_axis()` gains `toroidal`/`freeform` branches.
- `tests/`: new unit tests for the parser regex fix, the two new primitive extractions, the `elongated_boss` detection rule, and its operation classification.
- `freecad_addon/MachinaQCAM/`: no code changes required — the addon's correlation panel already surfaces whatever the core pipeline reports, same as the prior two changes in this area.

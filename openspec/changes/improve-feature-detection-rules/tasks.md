## 1. Parser-side primitive coverage

- [x] 1.1 In `src/parser.py`, populate `GeometryPrimitives.planes` from `PLANE` entities (replace the empty `pass` branch in `_extract_primitives`) with `(surface_id, normal, point)` resolved via new `AXIS2_PLACEMENT_3D`/`DIRECTION`/`CARTESIAN_POINT` resolvers, and populate `.cones` from `CONICAL_SURFACE` entities with `(surface_id, placement_ref, radius, semi_angle)`; verify with a unit test asserting `parser.primitives.planes` and `.cones` are non-empty, with resolved normal/point data, after parsing a STEP file containing planar/conical faces
- [x] 1.2 Expose the face-adjacency topology already computed by `_build_topology()`/`_classify_through_hole()` via a small accessor (e.g. `StepTextParser.get_face_adjacency() -> Dict[int, set]`), caching `_build_topology()`'s result so it isn't rebuilt per accessor call; verify with a unit test comparing the accessor's output against the topology maps built during hole detection on the same file
- [x] 1.3 Add `StepTextParser.get_face_bounding_extents(face_id, normal) -> Optional[Tuple[float, float]]`: resolve each face-bounding `EDGE_CURVE`'s `VERTEX_POINT`/`CARTESIAN_POINT` coordinates, project into the 2D plane orthogonal to `normal` via new `geometry.orthonormal_basis`/`bounding_extents_2d` helpers, and return `(long_extent, short_extent)`; verify with a unit test on a known rectangular face asserting the returned extents match its expected dimensions within tolerance

## 2. Pipeline wiring

- [x] 2.1 In `src/pipeline.py`, update `StepAnalyzer._extract_primitives()` to also build `SurfacePrimitive` entries for planar and conical primitives (not just cylindrical), and attach adjacency info from the new accessor to each `SurfacePrimitive`; verify with a unit test asserting the returned primitive list includes `planar`/`conical`-typed entries when the source file has them
- [x] 2.2 In `StepAnalyzer.analyze()`, build `hole`/`thread` features directly from `self.parser.features.holes` (through/blind, ASME/ISO label, standard, fit; a hole whose `asme_category == 'tap_drill'` is emitted as `thread` instead of `hole` per design.md's thread-evidence refinement) instead of via `FeatureDetector.detect_holes()`/`detect_threads()`; verify with a unit test asserting the report's `hole` features carry `is_through`/`asme_label` fields sourced from the parser, and a tap-drill-matched entry is emitted as `thread`
- [x] 2.3 Remove `FeatureDetector.detect_holes()` and `detect_threads()` and their call sites now that holes and threads come from the parser; verify by grepping the codebase for `detect_holes`/`detect_threads` and confirming no remaining references

## 3. Feature dataclass

- [x] 3.1 Add an `adjacent_face_ids` field to `SurfacePrimitive` in `src/primitive.py` (or wherever it's defined for pipeline use), defaulting to an empty list/None when unavailable; verify with a unit test constructing a `SurfacePrimitive` with and without adjacency data
- [x] 3.2 Establish the convention that every `Feature` returned by `FeatureDetector` includes a `"rationale": str` key in `parameters`; verify with a unit test asserting every `Feature` object produced by `detect_all_features` has a non-empty `parameters["rationale"]`

## 4. Rule tightening in `FeatureDetector`

- [x] 4.1 Rewrite `detect_slots()` to require narrow/elongated planar-pocket evidence (width/length ratio under a threshold, bounded by roughly-parallel edges via adjacency) instead of "is planar"; verify with unit tests covering both a qualifying slot geometry and a non-qualifying flat face (e.g. a large top face) per the spec's "Planar face with no supporting slot evidence" scenario
- [x] 4.2 Rewrite `detect_bosses()` to require exactly one adjacent planar face AND the face not already present in `parser.features.holes` (documented limitation: cannot yet distinguish true convex/concave without face-orientation data, see design.md 2a) instead of "is cylindrical or conical"; verify with a unit test covering a qualifying cylindrical face with one adjacent plane and not a matched hole, and one that is a matched hole (should not match)
- [x] 4.3 Thread evidence is handled in task 2.2 (pipeline-level, from `parser.features.holes`'s `asme_category`), not as a `FeatureDetector` method — see design.md's thread-evidence refinement; verify with the task 2.2 unit test plus a case with no tap-drill match, per the spec's "Cylindrical face with no thread evidence" scenario
- [x] 4.4 Rewrite `detect_drills()` to require a conical face adjacent to a compatible-radius cylindrical face (pilot-hole/countersink pattern) instead of "is conical"; verify with unit tests covering a qualifying countersink pattern and a standalone conical face that should not match
- [x] 4.5 Update `detect_all_features()` so a face matched by one rule is not also offered to subsequent rules, and collect any face matching no rule into an `unclassified_face_ids` list returned alongside the feature list; verify with a unit test asserting the "exactly one label" and "unclassified faces reported" scenarios from the spec

## 5. Report shape

- [x] 5.1 Update `StepAnalyzer.analyze()`'s returned report to include the `unclassified_face_ids` list from task 4.5; verify by asserting the key is present in `analyze()`'s output dict
- [x] 5.2 Update `README.md`'s feature-detection description (and any docs referencing per-face feature behavior) to describe exclusive, evidence-based classification instead of the previous type-only matching; verify by re-reading the updated section for accuracy against the shipped rules

## 6. Tests

- [x] 6.1 Add/update unit tests under `tests/` for `FeatureDetector` covering every scenario in `specs/manufacturing-feature-detection/spec.md`; verify with `pytest tests/` passing
- [x] 6.2 Add a regression test using at least one real file from `nist_sfa/` (or `nist_sfa/holeTrain/`) running the full `StepAnalyzer.analyze()` path end-to-end and asserting no face appears in more than one feature type's `face_ids`; verify with `pytest tests/` passing

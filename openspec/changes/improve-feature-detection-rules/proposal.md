## Why

`src/features.py`'s `FeatureDetector` — the rule-based classifier that turns geometric primitives into named manufacturing features (hole/boss/slot/thread/drill) — over-matches on primitive type alone: every planar face is currently labeled a `slot`, every cylindrical face with a positive radius is labeled a `thread`, and every cylindrical/conical face is labeled a `boss`, so a single part typically gets 3-4 conflicting feature labels per face. Separately, `src/pipeline.py`'s `StepAnalyzer._extract_primitives()` only ever extracts cylindrical primitives from the parser (`self.parser.primitives.cylinders`) — planar and conical primitives are never passed to `FeatureDetector` at all, so `detect_slots`/`detect_drills` are currently dead code in the STEP pipeline despite their rules existing. This makes feature output unreliable input for anything downstream, including the pending `add-cnc-operation-classifier` change, which consumes `Feature` objects to decide the required machining operation.

## What Changes

- Rewrite `FeatureDetector`'s per-feature rules to use additional geometric evidence beyond primitive type alone — face dimensions/proportions, and adjacency/topology already computed by `StepTextParser._build_topology()`/`_classify_through_hole()` (through vs. blind, adjacent-face counts) — so a face is assigned at most one feature label instead of multiple conflicting ones. **BREAKING**: a face that previously received several labels (e.g. both `slot` and `boss`) now receives one, changing the shape of `Feature` lists callers already consume.
- Fix `StepAnalyzer._extract_primitives()` to extract planar and conical primitives (not just cylindrical) from the parser output, so `detect_slots`/`detect_bosses`/`detect_drills` actually run on real data in the STEP pipeline instead of being unreachable.
- Reuse the parser's existing standards-validated hole detection (`StepTextParser.features.holes`, with through/blind classification and ASME/ISO size matching) as the authoritative source for `hole` features, instead of re-deriving holes independently and more crudely inside `FeatureDetector`.
- Add a confidence/rationale field to each detected `Feature` explaining which geometric evidence drove the label, so downstream consumers (including the operation classifier) can see why a face was classified a given way.

## Capabilities

### New Capabilities
- `manufacturing-feature-detection`: rule-based classification of STEP B-Rep geometric primitives into named manufacturing features (hole, boss, slot, thread, drill), each face assigned at most one feature and a rationale, using primitive type, dimensions, and face-adjacency topology as evidence.

### Modified Capabilities
- (none — `openspec/specs/` has no archived capabilities yet; this is this project's first feature-detection spec)

## Impact

- Modified: `src/features.py` (`FeatureDetector` rule bodies), `src/pipeline.py` (`StepAnalyzer._extract_primitives`, `_extract_primitives`'s call sites, and `analyze()`'s feature assembly to fold in parser-level hole data).
- Consumers of `Feature`/report JSON (e.g. `src/api.py` endpoints, `add-cnc-operation-classifier`'s planned `operation_classifier.py`) see a changed `features` list shape per part (fewer, non-overlapping features, each with a rationale) — noted as **BREAKING** above.
- No new dependencies; reuses existing `StepTextParser` topology/standards code already present for holes.
- Tests: `tests/test_pipeline.py` and any snapshot/expected-output tests asserting specific feature counts will need updating to match the corrected (smaller, non-duplicated) feature sets.

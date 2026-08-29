## Context

See proposal.md - Why. Relevant current state:

- `src/pipeline.py`'s `StepAnalyzer._extract_primitives()` only reads `self.parser.primitives.cylinders`; `self.parser.primitives.planes`/`.cones` are populated by `StepTextParser._extract_primitives()` (in `parser.py`) in principle but the dataclass field for planes is unused (`_extract_primitives` in `parser.py` has an empty `elif entity.type == 'PLANE': pass` branch) and cones aren't populated at all today — both need parser-side work, not just pipeline-side wiring.
- `src/features.py`'s `FeatureDetector` methods (`detect_holes`, `detect_bosses`, `detect_slots`, `detect_threads`, `detect_drills`) each independently scan the full primitive list and match on `primitive.type` alone (see proposal). `detect_all_features` concatenates all five, so overlap is by construction, not a bug in one method.
- `StepTextParser._detect_features()` (in `parser.py`) already does much better hole detection: standards-matched (ASME/ISO), through/blind via `_build_topology()`/`_classify_through_hole()`, stored in `self.features.holes`. This is currently unused by `FeatureDetector`/`pipeline.py` — `StepAnalyzer.analyze()` calls `self.detector.detect_all_features(primitives)` on the crude `SurfacePrimitive` list and never reads `self.parser.features.holes`.

## Goals / Non-Goals

**Goals:**
- Each face gets at most one feature label, using more than raw primitive type as evidence.
- `hole` features come from the parser's existing standards/topology logic, not a re-derivation.
- Planar and conical primitives reach `FeatureDetector` so its slot/boss/drill rules have real data to run on.
- Every feature (and, implicitly, every unclassified face) is explainable via a rationale string.

**Non-Goals:**
- No change to `StepTextParser`'s hole standards-matching/through-blind logic itself (decisions 1 doesn't touch `classify_hole`/`classify_hole_mm`/`_classify_through_hole`) — only how its output reaches `FeatureDetector`.
- No new ML model or learned classifier — rules stay hand-written and inspectable, matching the existing codebase's rule-based style and this change's rationale-reporting requirement.
- No attempt to detect features this change doesn't already list (e.g. pockets, chamfers, fillets) — scope is fixing the five existing feature types (hole, boss, slot, thread, drill), not expanding the taxonomy.
- Does not touch the pending `add-cnc-operation-classifier` change's own modules — that change consumes whatever `Feature` shape exists at the time it's implemented; this change's `## Impact` in proposal.md just flags the shape will differ.

## Decisions

**1. Threading `parser.features.holes` into the feature list happens in `pipeline.py`, not `features.py`.**
`StepAnalyzer.analyze()` already holds both `self.parser` (has `.features.holes`) and calls into `self.detector`. Add a step that converts `self.parser.features.holes` dicts into `Feature(feature_type="hole", face_ids=[hole['id']], parameters={...through/blind/asme fields..., "rationale": ...})` directly, and stop calling `FeatureDetector.detect_holes()` (delete it — it duplicated the parser's weaker independent logic per the proposal). Rationale: keeps the standards-matching/topology logic in the one place (`parser.py`) that already owns it; `features.py` stays about classifying the primitives the parser didn't already turn into named features (bosses/slots/threads/drills).

**2. `SurfacePrimitive` gains adjacency information, computed once by `PrimitiveClassifier`/pipeline, not by each `detect_*` method.**
Add an optional `adjacent_face_ids: List[int]` (or similar) field, populated from `StepTextParser`'s existing `face_edges`/`edge_to_faces` topology maps (already built for hole through/blind detection — expose via a small accessor rather than rebuilding). Individual `detect_*` rules read `primitive.details`/`adjacent_face_ids` rather than each re-deriving topology. Rationale: topology-building is the expensive/tricky part (see `_build_topology`'s complexity); computing it once and attaching it to each `SurfacePrimitive` avoids five independent implementations drifting apart.

**2a. Face-level geometry addendum (found necessary during implementation): real vertex/bounding-box extraction, not an adjacency-count proxy.**
The slot rule (dimensional evidence) and the boss rule (convex vs. concave) both need per-face geometry the parser didn't previously compute — `PLANE`/`CONICAL_SURFACE` entities were parsed for radius/semi-angle only, with no vertex, bounding-box, or normal-direction data. Per the user's explicit choice (over a coarser adjacency-count-only proxy), `parser.py` gains:
- Resolvers for `CARTESIAN_POINT`, `DIRECTION`, and `AXIS2_PLACEMENT_3D` entities (point + normal for a placement).
- `PLANE` primitives now carry `(surface_id, normal, point)`; `CONICAL_SURFACE` primitives carry `(surface_id, placement_ref, radius, semi_angle)` — both were previously unpopulated/absent (`PLANE` had an empty `pass` branch; `CONICAL_SURFACE` wasn't handled at all).
- `StepTextParser.get_face_bounding_extents(face_id, normal)`: resolves each `EDGE_CURVE` bounding a face to its `VERTEX_POINT`/`CARTESIAN_POINT` coordinates, projects them into the 2D plane orthogonal to the face's normal (via new `geometry.orthonormal_basis`/`bounding_extents_2d` helpers — generic vector math, not CNC-specific, so shared with `geometry.py` rather than duplicated), and returns `(long_extent, short_extent)`.
- `StepTextParser._build_topology()`'s result is now cached (`self._topology_cache`) since `get_face_adjacency()`, `get_face_bounding_extents()`, and `_detect_features()` all need it and it's the expensive part of parsing.
This is real geometric evidence (not a topology-count proxy): `slot` uses the true long/short extent ratio; `boss` still cannot perfectly distinguish convex from concave without a true B-Rep face-orientation flag (the simplified parser has no solid-angle/orientation data), so it uses one adjacent planar face **plus** "not already a parser-matched hole" as its evidence — documented as a known limitation in Risks/Trade-offs below, not silently treated as fully solved.

**3. Rule tightening per feature type, each gated on evidence beyond primitive type:**
- `slot`: planar face only when its long/short bounding-extent ratio (from `get_face_bounding_extents`) is at or above a narrow-pocket threshold, and it has a bounded number of adjacent faces consistent with a pocket (not a large exterior face) — replaces "is planar".
- `boss`: cylindrical/conical face only when it has exactly one adjacent planar face **and** is not already present in `parser.features.holes` (i.e. not already claimed as a standards-matched hole) — rather than "is cylindrical or conical". See 2a for why this can't yet distinguish true convexity.
- `thread`: **implementation refinement** — every standards-matched cylindrical face already becomes a `parser.features.holes` entry (parser's `_detect_features` claims any face that matches *any* ASME/ISO category, tap-drill included, before `FeatureDetector` ever sees it), so a cylindrical primitive that reaches `FeatureDetector` unclaimed can never carry thread evidence (it failed standards matching entirely). Thread evidence is therefore surfaced where the matched data already lives: in the pipeline's hole-feature construction (see decision 1), a hole entry whose `asme_category == 'tap_drill'` is emitted as a `thread` feature instead of a `hole` feature (a tap-drill-sized hole is the pre-thread state; the practical feature is "thread"), keeping the one-face-one-label guarantee. `FeatureDetector` has no `detect_threads()` method — there is nothing left for it to correctly detect once holes are excluded, and a method that can only ever return an empty list is dead code.
- `drill`: conical face only when adjacent to a cylindrical face of compatible radius (pilot-hole-with-countersink pattern) rather than "is conical".
- Faces satisfying none of the above are left unclassified (spec's "Unclassified faces" requirement) — `detect_all_features` returns the union of matched features plus, in the returned dict/report (not the `Feature` list itself), an `unclassified_face_ids` list for visibility.
Rationale: minimal, targeted tightening of each existing rule using data already available from the parser/topology work in decision 2, rather than a rule-engine rewrite — keeps the change reviewable and matches the proposal's stated scope.

**4. `Feature.parameters` gains a `rationale: str` key; no new dataclass field.**
Rationale: `Feature` is a small, already-used dataclass (`feature_type`, `face_ids`, `parameters`); adding the string into the existing free-form `parameters` dict avoids touching every construction site's positional/keyword shape while still satisfying the spec's rationale requirement, and mirrors how `hole_info` (in `parser.py`) already carries many such descriptive fields in a dict.

## Risks / Trade-offs

- **[Risk]** Tightening slot/boss/thread/drill rules (decision 3) means some previously-flagged (if often wrong) features disappear entirely rather than being relabeled — a consumer relying on "at least something gets flagged per face" will see fewer features. → **Mitigation**: this is the explicit intent (proposal's **BREAKING** note); the new `unclassified_face_ids` list keeps that information visible instead of silently vanishing.
- **[Risk]** The boss rule (decision 2a/3) still cannot distinguish a convex protrusion from a concave pocket with identical adjacency topology (one adjacent planar face), since the simplified text parser has no true face-orientation/solid-angle data. → **Mitigation**: excluding anything already matched as a standards-validated hole (`parser.features.holes`) removes the most common false positive (blind holes); documented as a known accuracy limit, not claimed as solved — a candidate for a future change if real B-Rep orientation data is added.
- **[Risk]** Thread detection now requires pitch/designation data that may not exist for all cylindrical faces in files without thread callouts, so `thread` features may become rare or absent on files that previously (wrongly) flagged many threads. → **Mitigation**: acceptable — a `thread` label with no supporting evidence was actively misleading; fewer, correct labels is the goal per the spec's "no thread evidence → not classified as thread" scenario.
- **[Trade-off]** Computing adjacency once per `SurfacePrimitive` (decision 2) adds a small amount of topology work to every analysis run (previously only paid for hole through/blind checks) → acceptable, since `_build_topology()` already runs once per `analyze()` call regardless.

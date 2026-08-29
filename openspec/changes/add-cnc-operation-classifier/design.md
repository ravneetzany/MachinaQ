## Context

See proposal.md - Why. Relevant current state:

- `src/primitive.py` classifies STEP B-Rep faces into `SurfacePrimitive(face_id, type, details)` where `type` is `planar` | `cylindrical` | `conical` | `unknown`.
- `src/features.py`'s `FeatureDetector` turns a list of `SurfacePrimitive` into `Feature(feature_type, face_ids, parameters)` (hole/boss/slot/thread/drill), purely from primitive `type` — it has no notion of a part's principal axis today.
- `src/pipeline.py`'s `StepAnalyzer.analyze()` is the single orchestration point: parse → primitives → features → (optional) PointNet predictions → JSON report.
- `/home/ravneetzany/projects/openscad-parts-library` and `/freecad-parts-library` are parametric generator sources (not exported models). Their comments already state the intended process for several parts (`bushing.scad`: "CNC-turned"; `l_bracket.scad`: "CNC-milled"; `stepped_shaft.scad`: "CNC-turned ... with a keyway"), which makes them a convenient labeled smoke-test set, not just raw geometry input.
- FreeCAD `.py` scripts under `freecad-parts-library` import `FreeCAD`/`FastenersCmd` at module scope and require FreeCAD's bundled Python + addon libraries to run — invoking them means a `freecadcmd` subprocess and an installed FreeCAD 1.1 with the Fasteners/gears addons, which is heavier than this change needs. Per the user's decision, ingestion parses source statically instead of executing these scripts.

## Goals / Non-Goals

**Goals:**
- Classify each existing `Feature` (and the part's base body) with a required CNC operation, using only information derivable from primitive type + a per-part principal-axis determination.
- Add a source path (`.scad` / FreeCAD `.py`) that produces `SurfacePrimitive`-shaped output, so `FeatureDetector` and the new classifier work unmodified on either STEP-derived or source-derived input.
- Batch-run over a directory of parts and emit one JSON report per part.

**Non-Goals:**
- No CAD kernel execution (no `freecadcmd`, no OpenSCAD binary invocation) — parsing is static/textual.
- No feed/speed/tool selection, no G-code, no cost estimation.
- No change to the STEP ingestion path's behavior (`StepTextParser`, `PrimitiveClassifier` stay as-is).
- No handling of arbitrary/general OpenSCAD or Python (only the primitive-call vocabulary actually used by these two libraries: `cyl`/`cylinder`, `cube`/`cuboid`, `translate`, `rotate`, `union`, `difference`, and the libraries' own `make_fastener`/`make_gear`/similar wrapper calls).

## Decisions

**1. Axis/operation model lives in a new module, not inside `FeatureDetector`.**
`src/operation_classifier.py` takes `(primitives: List[SurfacePrimitive], features: List[Feature], principal_axis: Optional[Axis])` and returns per-feature operations + a part rollup. Rationale: keeps `FeatureDetector`'s existing STEP-path behavior untouched (proposal explicitly says the STEP pipeline is unchanged), and keeps axis reasoning — which only the ingestion layer can compute cheaply from source, and which STEP primitives don't currently carry — as an explicit optional input rather than inferred from face lists alone.

**2. `principal_axis` is a new, optional field the STEP path leaves `None`.**
For STEP input, `PrimitiveClassifier` does not currently expose a part-wide axis, so the classifier degrades gracefully (falls back to feature-type-only heuristics: cylindrical→turning-candidate, planar→milling-candidate, no axis-coaxiality refinement, more `unknown`/lower-confidence rationale text) rather than blocking. For `.scad`/`.py` ingestion (design decision 3), the ingester computes it directly from primitive transforms, giving full-fidelity classification for the corpus this change targets.

**3. Ingestion is a static parser per language, both emitting the same intermediate shape.**
- `src/scad_ingest.py`: regex/tokenizer-based extraction of the primitive calls listed in Non-Goals from a `.scad` file's top-level module body (not a full OpenSCAD-language implementation — unsupported constructs are recorded as unparsed rather than guessed). Tracks a running transform stack through `translate`/`rotate` to compute each primitive's absolute axis, then reduces to a single `principal_axis` when all cylindrical primitives share one axis.
  - **Vocabulary addendum (found necessary during implementation):** `stepped_shaft.scad` — one of the three labeled corpus files this change's tasks require parsing — builds its segment layout with an OpenSCAD C-style `for` loop and a list comprehension (`[for (i=0, acc=0; i<len(segments); acc=acc+segments[i][1], i=i+1) acc]`). The parser additionally supports this one general pattern: a numeric `for (init; cond; update) expr` loop over `init`/`cond`/`update` clauses using only variable assignment, `len()`, array indexing, and `+`/`-` arithmetic on bound variables, evaluated to produce a list. Any `for`/list-comprehension shape outside that (nested loops, other function calls, string/boolean accumulation) is still reported as unparsed rather than guessed — this is a narrow, general-enough extension of the supported vocabulary, not a `stepped_shaft.scad`-specific special case.
- `src/py_source_ingest.py`: uses Python's `ast` module to statically find `make_*(...)` call nodes and evaluate their literal/keyword arguments (`ast.literal_eval`-safe subset only) without importing the module — this is what lets fastener/gear scripts be read without a FreeCAD runtime. Each wrapper function (`make_fastener`, `make_gear`, ...) maps to a small, explicit geometry template (bolt/nut → cylindrical shaft primitive; spur/bevel/worm gear → cylindrical disc + non-analyzed tooth features) maintained alongside the parser, since the actual FreeCAD geometry isn't being computed.
- Both output `List[SurfacePrimitive]` plus the derived `principal_axis`, so they plug into the existing `FeatureDetector.detect_all_features()` unchanged, satisfying the parametric-source-ingestion spec's "output compatible with existing feature representation" requirement.

**4. Batch CLI is a thin new script, not a `pipeline.py` rewrite.**
A new `scripts/classify_directory.py` (or `src/batch_classify.py`) walks a directory, picks the ingester by extension, and calls `StepAnalyzer`-equivalent logic (reusing `FeatureDetector` + the new classifier) per file, writing `<part-name>.json` per part. `pipeline.py`'s `StepAnalyzer.analyze()` gains the classifier call (feature → operation, plus rollup) so STEP-sourced reports get the same `operation`/`operations_summary` fields; the batch script is additive and doesn't change `StepAnalyzer`'s existing STEP-only entry point.

**5. Axis-coaxiality tolerance and the turning-vs-5-axis boundary are simple, explainable geometric rules, not a learned model.**
E.g. "coaxial" = same axis direction (within a small angular tolerance) and axis line passing through the same point (within a small positional tolerance) as the part's dominant cylindrical primitive; "reachable along 3 orthogonal axes" = feature's local normal/axis aligns with one of the part's X/Y/Z after accounting for accumulated rotate transforms, else 5-axis. Rationale: matches the spec's rationale-reporting requirement (a rule can state *why* in plain English; a black-box classifier can't), and the labeled comments in the source corpus (`bushing.scad`, `l_bracket.scad`, `stepped_shaft.scad`) are few enough to validate rules against directly rather than needing to train anything.

## Risks / Trade-offs

- **[Risk]** Static `.scad`/`.py` parsing only covers the primitive vocabulary actually observed in these two libraries; a differently-authored part script (e.g. using `hull()`, loops building primitives dynamically, or a wrapper function outside the known `make_*` set) will be reported as unparsed. → **Mitigation**: the spec requires unparsed files to be reported with a reason, not silently skipped or crashed on; expand the known-primitive/known-wrapper tables incrementally as new scripts are encountered.
- **[Risk]** FreeCAD wrapper geometry templates (decision 3) are hand-maintained approximations, not the real generated `Shape` — a gear's actual involute tooth geometry isn't modeled, only "disc, axisymmetric." → **Mitigation**: acceptable for this change's goal (operation-type classification, which for gears/fasteners is turning/milling at the disc-body level regardless of tooth detail); explicitly called out as a non-goal to model exact tooth geometry.
- **[Risk]** The STEP path's `principal_axis: None` fallback (decision 2) means STEP-derived parts get coarser classification (more `unknown`/generic milling) until `PrimitiveClassifier` is extended to compute a part-wide axis from B-Rep data — out of scope here. → **Mitigation**: explicitly scoped as a fallback, not a bug; future change can extend `PrimitiveClassifier` if STEP-path parity is needed.
- **[Trade-off]** Choosing static parsing over executing `freecadcmd`/`openscad` (per the user's answer) trades geometric accuracy for zero new runtime dependencies and no need for a FreeCAD/OpenSCAD install in CI or on the classifying machine.

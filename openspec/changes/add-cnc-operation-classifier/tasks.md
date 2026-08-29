## 1. Shared data model

- [x] 1.1 Add an `Axis` type (direction vector + point) and an optional `principal_axis` field to the data passed between ingestion and the classifier; verify with a unit test constructing an `Axis` and comparing two coaxial axes within tolerance
- [x] 1.2 Define the `Operation` enum/constants (`turning`, `drilling`, `face_milling`, `3_axis_milling`, `5_axis_milling`, `unknown`) in `src/operation_classifier.py`; verify by importing the module and asserting all six values exist

## 2. OpenSCAD ingestion (`src/scad_ingest.py`)

- [x] 2.1 Implement a tokenizer/parser for the primitive call vocabulary (`cyl`/`cylinder`, `cube`/`cuboid`, `translate`, `rotate`, `union`, `difference`) that walks a `.scad` file's top-level module body and emits `SurfacePrimitive`-shaped records with an absolute transform; verify against `bushing.scad` and `stepped_shaft.scad`, asserting the expected cylindrical primitives are extracted
- [x] 2.2 Implement principal-axis reduction: when all cylindrical primitives share one axis (within tolerance), report a single `principal_axis`; otherwise report `None`; verify with `bushing.scad` (single axis) and `l_bracket.scad` (no single axis, two perpendicular plates)
- [x] 2.3 Implement directory discovery that walks a library root, yields `.scad` files outside `lib/`, `docs/`, `tests/`, and skips/report files it cannot parse (unsupported constructs) with a reason instead of raising; verify by running discovery against `/home/ravneetzany/projects/openscad-parts-library` and asserting no unhandled exception and at least one parsed and zero silently-dropped files

## 3. FreeCAD source ingestion (`src/py_source_ingest.py`)

- [x] 3.1 Implement an `ast`-based static extractor that finds `make_*(...)` call nodes in a `.py` file and resolves literal/keyword arguments via `ast.literal_eval` (no import of the script or FreeCAD); verify against `fastener.py`'s `make_fastener("ISO4014", diameter="M8", length="30")` call, asserting the arguments are recovered without importing `FreeCAD`
- [x] 3.2 Add a small geometry-template table mapping known wrapper functions (`make_fastener`, `make_gear`, and any others present in the library) to primitive shapes (e.g. bolt/nut → cylindrical shaft, gear → cylindrical disc), each producing `SurfacePrimitive`-shaped output plus `principal_axis`; verify with a unit test per mapped wrapper function
- [x] 3.3 Implement directory discovery that walks a library root, yields `.py` files (outside `lib/`) defining a recognized `make_*` call, and reports files with unresolvable/dynamic arguments as unparsed with a reason instead of raising; verify by running discovery against `/home/ravneetzany/projects/freecad-parts-library` and asserting `lib/common.py` is excluded and no unhandled exception occurs

## 4. Operation classifier (`src/operation_classifier.py`)

- [x] 4.1 Implement feature-level classification rules (coaxial-cylindrical → turning; planar on non-rotational part → face/3-axis milling; off-axis hole → drilling; non-orthogonal/non-coaxial → 5-axis milling; missing/unknown primitive type → unknown) per the `cnc-operation-classification` spec scenarios; verify with a unit test per scenario in `specs/cnc-operation-classification/spec.md`
- [x] 4.2 Implement the STEP-path fallback (`principal_axis is None`) using feature-type-only heuristics with a rationale noting the coarser confidence; verify with a unit test feeding STEP-derived `SurfacePrimitive`/`Feature` objects (no axis) through the classifier and asserting it returns a result, not an error
- [x] 4.3 Implement the part-level rollup (primary process by feature count / base-body operation, secondary processes for the rest, `unknown` features excluded from the vote) with per-item rationale strings; verify with a unit test covering the "single operation" and "turned body with non-coaxial secondary feature" scenarios from the spec
- [x] 4.4 Wire per-feature `operation` and part-level `operations_summary` into `StepAnalyzer.analyze()`'s returned report in `src/pipeline.py`; verify by running the existing STEP-based analyzer test/flow and asserting the JSON report now includes `operation` per feature and an `operations_summary` block

## 5. Batch CLI

- [x] 5.1 Add a batch script (e.g. `scripts/classify_directory.py`) that walks a given directory, dispatches to `scad_ingest` or `py_source_ingest` by extension, runs `FeatureDetector` + `operation_classifier`, and writes one JSON report per part; verify by running it against `/home/ravneetzany/projects/openscad-parts-library` and `/home/ravneetzany/projects/freecad-parts-library` and confirming a report file is produced per discovered part with no unhandled exception
- [x] 5.2 Spot-check classifier output against the source comments that already state intended process (`bushing.scad` → turning, `l_bracket.scad` → milling, `stepped_shaft.scad` → turning with a milled keyway) and record any mismatches; verify by listing the reports for these three parts and confirming the reported primary/secondary operations match the comment-stated process (or documenting why not, if a genuine rule gap is found)
  - Result: `bushing.scad` → turning ✓, `stepped_shaft.scad` → turning primary + 3-axis-milling secondary (keyway) ✓, `l_bracket.scad` → drilling primary + 3-axis-milling secondary ✗ (comment says milling). Spot-checking this file surfaced and fixed a real bug (the no-principal-axis fallback was misclassifying cylindrical features as `turning`, contradicting the approved spec — fixed, see design.md decision 2's "Correction found during implementation"). The remaining `l_bracket.scad` mismatch is a separate, narrower rule gap (no base-body-vs-feature-count priority rule for prismatic parts) documented in design.md's Risks/Trade-offs rather than fixed here, per this task's own allowance.

## 6. Tests and docs

- [x] 6.1 Add unit tests under `tests/` for `scad_ingest`, `py_source_ingest`, and `operation_classifier` covering every scenario in both delta specs; verify with `pytest tests/` passing
- [x] 6.2 Update `README.md`'s architecture section and endpoint/feature list to mention operation classification and the new batch entry point; verify by re-reading the updated section for accuracy against the shipped module names

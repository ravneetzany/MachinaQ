## Why

MachinaQ's pipeline detects and labels machining *features* (holes, bosses, slots, threads, drills) but never says which *process* (turning, milling, drilling, 5-axis) is required to cut them. Estimators and CAM planners still have to eyeball each part to route it. `/home/ravneetzany/projects/` holds two parametric CAD libraries (`freecad-parts-library`, `openscad-parts-library`) whose generator scripts (`.py` / `.scad`) are a ready-made, labeled-in-comments test corpus (e.g. `bushing.scad` is explicitly "CNC-turned", `l_bracket.scad` is explicitly "CNC-milled") for building and validating this classifier without needing a CAD kernel or exported STEP files.

## What Changes

- Add a new **operation classifier** module that assigns a required CNC operation (Turning, Drilling, Face Milling, 3-Axis Milling, 5-Axis Milling) to each detected geometric feature, plus a part-level rollup summarizing the primary and secondary processes needed.
- Add a **parametric-source ingestion path** that parses OpenSCAD (`.scad`) and FreeCAD wrapper-script (`.py`) parts from a directory into the same lightweight primitive/feature representation the existing STEP pipeline uses, so the classifier (and existing `FeatureDetector`) can run without a STEP export step. This is additive — the existing `StepTextParser` → `PrimitiveClassifier` → `FeatureDetector` STEP path is unchanged.
- Extend the JSON report schema with an `operation` field per feature and an `operations_summary` block per part (primary process, secondary processes, rationale per feature).
- Add a CLI/script entry point that batch-scans a directory of `.scad` / FreeCAD `.py` part scripts (e.g. `/home/ravneetzany/projects/*-parts-library`), runs ingestion + classification, and writes one JSON report per part.
- **Not included**: exporting real STEP/STL geometry via headless FreeCAD/OpenSCAD execution, CAM toolpath generation, and machine/tool selection — this change only classifies the *type* of operation, not feeds/speeds/tooling.

## Capabilities

### New Capabilities
- `cnc-operation-classification`: given a part's detected geometric features (or its whole-part shape), determine the required CNC operation type(s) per feature and an overall part-level process recommendation, with a stated rationale.
- `parametric-source-ingestion`: parse OpenSCAD (`.scad`) and FreeCAD wrapper-script (`.py`) part definitions from a directory into MachinaQ's existing primitive/feature representation, as an alternative input path to STEP files.

### Modified Capabilities
- (none — no existing spec files; the STEP-based parsing/feature-detection code is extended via a new module, not by changing its behavior)

## Impact

- New code: `src/operation_classifier.py`, `src/scad_ingest.py` (or similarly named ingestion module), a batch CLI script.
- Modified: `src/pipeline.py` (wire the classifier into `StepAnalyzer.analyze` output), `src/api.py` (if the report schema is exposed via the existing FastAPI endpoints), report JSON schema/consumers.
- New dependency: none required for `.scad` parsing (regex/AST-lite parser); FreeCAD `.py` scripts import `FreeCAD` at module scope, which requires FreeCAD's Python environment to execute directly — ingestion instead statically parses the wrapper call arguments rather than executing the scripts, avoiding a FreeCAD runtime dependency.
- Data: `/home/ravneetzany/projects/freecad-parts-library` and `/home/ravneetzany/projects/openscad-parts-library` become an external, read-only input source for testing/validating the classifier.

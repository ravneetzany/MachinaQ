## Why

MachinaQ can already classify a STEP file's features and required CNC operations end-to-end (`StepAnalyzer.analyze()`, exposed via `POST /analyze`), but using it today means manually exporting a STEP file and calling the API or CLI outside of FreeCAD. CAM programmers work inside FreeCAD's CAM workbench, selecting geometry and setting up operations by hand — there is no way to ask "what feature is this, and what operation does MachinaQ think it needs" without leaving FreeCAD. This change adds that as a button inside the CAM workbench.

**Note on scope decisions:** the user's request had several points that would materially change what gets built (how FreeCAD talks to MachinaQ; how the button gets into CAM's toolbar; how a FreeCAD-selected face maps back to a MachinaQ-reported feature) and the user declined the clarifying questions asked, choosing instead for reasonable assumptions to be made and recorded here for review. Every assumption below is a specific, revisable choice — flag any of them to change direction before `/opsx:apply`.

## What Changes

- **Assumption — integration path: HTTP client, not embedded MachinaQ.** The FreeCAD-side code calls MachinaQ's existing `POST /analyze` endpoint over HTTP (assumes the API server is already running and reachable, e.g. `uvicorn src.api:app`), rather than importing `src`/`models` directly into FreeCAD's own Python interpreter. Rationale: FreeCAD ships its own embedded Python, separate from MachinaQ's `.venv`; installing `torch` and the rest of MachinaQ's dependency set into FreeCAD's interpreter is heavy and version-fragile (FreeCAD's bundled Python version/ABI may not match available `torch` wheels), whereas an HTTP call only needs the standard library (`urllib`) or `requests`, which FreeCAD's Python already has in most distributions.
- **Assumption — same-machine deployment only, for now.** `POST /analyze` takes a server-side `step_path`, not an uploaded file — it assumes the API server can read the same filesystem the caller writes to. This change keeps that constraint (FreeCAD exports its temp STEP file to a path the MachinaQ server also reads locally); a remote/multipart-upload API is explicit non-goal, noted for a future change if needed.
- Add a FreeCAD macro/addon package (`freecad_addon/MachinaQCAM/`) that, on FreeCAD startup, registers a new command (`MachinaQ_ClassifyFeature`) and adds it into the CAM workbench's existing toolbar/command list — not a new standalone workbench.
- The command, when activated: exports the active document's selected Body (or the whole active document if nothing is selected) to a temporary STEP file, calls `POST /analyze` against a configurable MachinaQ API URL (default `http://127.0.0.1:8000`), and displays the resulting feature list + per-feature operation + part-level `operations_summary` in a FreeCAD task panel.
- **Best-effort face correlation, not exact matching.** If the user has one or more specific Faces selected (not just a Body), the command highlights which reported feature(s) are geometrically nearest to the selected face(s) — using each face's center-of-mass compared against the nearest MachinaQ-reported primitive's position (a point, for planar primitives; the closest point on the axis line, for cylindrical/conical primitives). This is an approximation, not an exact face-id correspondence (STEP re-parsing and FreeCAD's own face indexing are not guaranteed to agree), and is reported to the user as such — see design.md for why exact matching isn't attempted.
- Add cylindrical/conical primitive axis-position extraction to `src/parser.py` (currently only planar primitives carry a resolved 3D point; cylinders/cones do not) — a small, necessary prerequisite for the face-correlation matching above to work for holes/bosses, not just flat faces.
- **Not included**: a distributed/remote MachinaQ deployment (multipart upload API), exact (non-approximate) face-to-feature correspondence, and automatically applying the recommended operation to a FreeCAD CAM operation object (this change only *reports* the recommendation; acting on it — e.g. auto-creating a Drilling/Profile operation — is a natural follow-up, explicitly out of scope here).

## Capabilities

### New Capabilities
- `freecad-cam-integration`: a FreeCAD CAM-workbench command that exports the active document to MachinaQ, displays its feature/operation classification, and best-effort correlates the result back to the user's current face selection.

### Modified Capabilities
- (none — this change adds a new external-integration surface; it does not change any existing MachinaQ spec's behavior, though it does extend `src/parser.py`'s primitive extraction, a prerequisite implementation detail rather than a behavior change to any existing requirement)

## Impact

- New: `freecad_addon/MachinaQCAM/` (InitGui.py, the command class, an icon, a task-panel UI), likely a small HTTP client helper.
- Modified: `src/parser.py` (cylinder/cone primitives gain a resolved axis position, reusing the `_resolve_placement` helper already built for planes).
- No change to `POST /analyze`'s existing request/response shape — the FreeCAD side is purely a new consumer of it.
- Deployment: requires the user to have MachinaQ's API server running and reachable from the machine FreeCAD runs on, and the addon installed into FreeCAD's `Mod/` (or `Macro/`) directory — this change does not build a FreeCAD Addon Manager listing/installer, just the addon package itself.

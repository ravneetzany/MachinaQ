# MachinaQCAM

A FreeCAD addon that adds a "Classify Feature (MachinaQ)" command to the
**CAM workbench**'s toolbar. Activating it exports the active document (or
your selected Body) to a temporary STEP file, submits it to a running
MachinaQ API server, and shows the resulting per-feature machining-operation
classification (turning / drilling / face milling / 3-axis / 5-axis milling)
in a read-only panel.

## Prerequisites

**The MachinaQ API server must already be running and reachable** from the
machine FreeCAD runs on — this addon is an HTTP client, not an embedded copy
of MachinaQ (see `../../openspec/changes/add-freecad-cam-integration/design.md`
decision 1 for why). Start it from the MachinaQ repo:

```bash
uvicorn src.api:app --reload
```

By default the addon talks to `http://127.0.0.1:8000`. Change it via
FreeCAD's `User parameter:BaseApp/Preferences/Mod/MachinaQCAM/ApiUrl` if
your server runs elsewhere reachable on the same network — the API itself
only accepts a server-side file path (`step_path`), so both FreeCAD and the
MachinaQ server need access to the same filesystem where the temp STEP file
is written; this addon does not support a fully remote deployment.

## Installation

Copy or symlink this directory into FreeCAD's `Mod/` directory, e.g.:

```bash
ln -s /home/ravneetzany/MachinaQ/freecad_addon/MachinaQCAM ~/.local/share/FreeCAD/Mod/MachinaQCAM
```

Restart FreeCAD, switch to the **CAM** workbench, and the "Classify Feature
(MachinaQ)" button appears in a new "MachinaQ" toolbar group.

## Usage

1. Open a document in FreeCAD, switch to the CAM workbench.
2. Optionally select a Body (otherwise the whole document is exported), or
   select one or more Faces to also get a face-correlation section.
3. Click "Classify Feature (MachinaQ)".
4. A panel shows the part's primary/secondary machining process, every
   detected feature's type and operation, and — if you selected specific
   faces — which classified feature each one is nearest to.

## Important limitation: face correlation is approximate, not exact

FreeCAD's own `Shape.Faces` indexing and MachinaQ's STEP-parsed `face_id`
are not guaranteed to correspond to the same face. The face-correlation
section instead reports the **nearest classified feature by geometric
position** (closest point for flat faces, closest point on the axis line
for cylindrical/conical faces) — this can occasionally pick the wrong one
among several similar, closely-spaced features (e.g. a bolt pattern of
identical holes). It is always labeled as an approximate match in the
panel; treat it as a hint, not a guaranteed answer.

## What this addon does NOT do

- It does not create, modify, or configure any FreeCAD CAM operation —
  it only reports MachinaQ's recommendation.
- It does not work with a remote MachinaQ server on a different filesystem.
- It is not published to FreeCAD's Addon Manager — install manually as above.

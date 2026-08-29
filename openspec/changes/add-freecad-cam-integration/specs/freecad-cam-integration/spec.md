## Purpose

Let a CAM programmer working inside FreeCAD's CAM workbench trigger MachinaQ's feature/operation classification on the active document without leaving FreeCAD, and see a best-effort correlation between their current face selection and MachinaQ's reported features.

## ADDED Requirements

### Requirement: Command registered into the CAM workbench
The addon SHALL register a command into FreeCAD's CAM workbench's toolbar/command list when FreeCAD starts, so it is available whenever the CAM workbench is active, without the user having to build a custom toolbar themselves.

#### Scenario: Command appears in the CAM workbench
- **WHEN** FreeCAD starts with the addon installed and the user switches to the CAM workbench
- **THEN** a "MachinaQ: Classify Feature" command is present in the CAM workbench's toolbar/command list

### Requirement: Export and classify the active document
Activating the command SHALL export the relevant geometry (the selected Body if one is selected, otherwise the whole active document) to a temporary STEP file and request classification from MachinaQ's `POST /analyze` endpoint at a configurable API URL.

#### Scenario: Body selected
- **WHEN** the user has a Body selected in the active document and activates the command
- **THEN** that Body is exported to a temporary STEP file and submitted for classification

#### Scenario: Nothing selected
- **WHEN** the user has no selection and activates the command
- **THEN** the whole active document is exported to a temporary STEP file and submitted for classification

### Requirement: Display feature and operation results
The command SHALL display the classification response's feature list (each feature's type and its per-feature `operation`) and the part-level `operations_summary` (primary process, secondary processes, rationale) in a FreeCAD task panel.

#### Scenario: Successful classification
- **WHEN** MachinaQ returns a classification report
- **THEN** the task panel shows each feature's type and operation, plus the part-level primary/secondary process summary and its rationale

### Requirement: Best-effort face correlation, explicitly labeled as approximate
When one or more specific Faces (not just a Body) are selected, the command SHALL report which classified feature(s) are geometrically nearest to each selected face, using a position-based nearest-match (a point for planar primitives, the nearest point on the axis line for cylindrical/conical primitives), and SHALL label this correlation as approximate rather than presenting it as an exact match.

#### Scenario: Face selected, nearest feature reported
- **WHEN** the user has one or more Faces selected and activates the command
- **THEN** the task panel reports, for each selected face, the nearest classified feature and its operation, with a visible note that the match is approximate (not a guaranteed exact correspondence)

### Requirement: Graceful handling of an unreachable or erroring API
If the MachinaQ API server is unreachable, times out, or returns an error response, the command SHALL show a clear error message in FreeCAD rather than crash FreeCAD or fail silently.

#### Scenario: API unreachable
- **WHEN** the configured MachinaQ API URL cannot be reached
- **THEN** the command shows an error message stating the API could not be reached, and FreeCAD remains fully usable afterward

#### Scenario: API returns an error
- **WHEN** MachinaQ's API responds with a non-success status (e.g. the exported STEP file fails to parse)
- **THEN** the command shows the returned error detail rather than a generic or silent failure

### Requirement: Cylindrical and conical primitives carry a resolved position
`StepTextParser` SHALL resolve and expose a 3D axis position for cylindrical and conical primitives (not only planar primitives, which already have one), so face-correlation matching (see above) has position data to compare against for hole/boss-type features, not only flat faces.

#### Scenario: Cylindrical primitive position available
- **WHEN** a STEP file containing a cylindrical face is parsed
- **THEN** the resulting primitive's data includes a resolved axis position (point and direction), not only its radius

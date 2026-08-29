## ADDED Requirements

### Requirement: Elongated boss detection from toroidal and free-form surfaces
The system SHALL classify a `toroidal`, `freeform`, or `cylindrical` primitive not already claimed by hole, thread, boss, slot, or planar_face detection as an `elongated_boss` feature when its face's bounding-extent aspect ratio meets an elongation threshold, so curved protrusions (e.g. a swept or lofted rib/arm) become eligible for operation classification rather than being silently dropped or reported as unclassified. This feature type is named `elongated_boss`, not an operation name, for the same reason `planar_face` is not named `face_milling` — a detected geometric feature and a required-operation determination are separate concepts and must not share a name.

#### Scenario: Elongated toroidal or free-form face is classified as an elongated_boss
- **WHEN** a `toroidal` or `freeform` primitive's face is not claimed by any hole, thread, boss, slot, or planar_face rule, and its bounding-extent aspect ratio meets the elongation threshold
- **THEN** the system's output contains an `elongated_boss` feature entry referencing that face's `face_id`, with a rationale describing the extent evidence used

#### Scenario: elongated_boss feature reaches operation classification
- **WHEN** an `elongated_boss` feature has been detected
- **THEN** it is included in the feature list passed to operation classification, so it receives a required-operation determination (3-axis milling or 5-axis milling) rather than being silently excluded or reported as unknown

### Requirement: Toroidal surface primitive extraction
The system SHALL extract `TOROIDAL_SURFACE` STEP entities into a `toroidal` geometric primitive carrying axis position/direction data (via the same placement-resolution mechanism used for cylindrical and conical surfaces), so toroidal faces are available to feature detection.

#### Scenario: Toroidal primitives reach feature detection
- **WHEN** a STEP file's parsed geometry includes toroidal surfaces
- **THEN** those toroidal primitives are present in the primitive list passed into feature detection, each carrying resolvable axis data

### Requirement: Free-form (B-spline) surface primitive extraction
The system SHALL extract `B_SPLINE_SURFACE_WITH_KNOTS` STEP entities into a `freeform` geometric primitive carrying the face's bounding extents (not exact NURBS control-point/knot-vector data), so free-form faces are available to feature detection despite the system not modeling their exact analytic geometry.

#### Scenario: Free-form primitives reach feature detection
- **WHEN** a STEP file's parsed geometry includes B-spline surfaces
- **THEN** those free-form primitives are present in the primitive list passed into feature detection, each carrying resolvable bounding-extent data when the underlying face has resolvable boundary vertices

### Requirement: Multi-line STEP entities are parsed, not silently dropped
The system SHALL parse a STEP entity definition regardless of whether its attribute list spans multiple lines in the source file.

#### Scenario: A multi-line entity is present in the parsed entity table
- **WHEN** a STEP file contains an entity (of any type) whose attribute list spans multiple lines before its terminating semicolon
- **THEN** that entity is present in the parser's entity table with its correct id and type, not silently omitted

## MODIFIED Requirements

### Requirement: Full primitive coverage
The system SHALL make planar, cylindrical, conical, toroidal, and free-form (B-spline) primitives available to feature detection, so rules that depend on any of these primitive types can actually execute.

#### Scenario: Planar primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes planar surfaces
- **THEN** those planar primitives are present in the primitive list passed into feature detection (not silently dropped before detection runs)

#### Scenario: Conical primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes conical surfaces
- **THEN** those conical primitives are present in the primitive list passed into feature detection

#### Scenario: Toroidal primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes toroidal surfaces
- **THEN** those toroidal primitives are present in the primitive list passed into feature detection

#### Scenario: Free-form primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes B-spline surfaces
- **THEN** those free-form primitives are present in the primitive list passed into feature detection

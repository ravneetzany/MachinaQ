## Purpose

Determine which CNC machining operation (turning, drilling, or a milling variant) is required to produce each detected feature of a part, and roll that up into a part-level process recommendation, so downstream CAM/estimation workflows don't have to eyeball geometry manually.

## ADDED Requirements

### Requirement: Per-feature operation classification
The system SHALL assign one required CNC operation type — one of `turning`, `drilling`, `face_milling`, `3_axis_milling`, or `5_axis_milling` — to each detected feature, based on the feature's geometric primitive type and its relationship to the part's principal axis (if any).

#### Scenario: Cylindrical feature coaxial with the part's single rotational axis
- **WHEN** a feature's primitive is cylindrical and coaxial with the part's single, part-wide principal rotational axis
- **THEN** the system classifies the feature as `turning`

#### Scenario: Cylindrical or conical hole not coaxial with the principal axis
- **WHEN** a feature is a hole/drill-type feature whose axis is parallel to a flat face's normal but not coaxial with the part's principal rotational axis (or the part has no single principal axis)
- **THEN** the system classifies the feature as `drilling`

#### Scenario: Planar feature on a prismatic (non-rotational) part
- **WHEN** a feature's primitive is planar and the part has no single principal rotational axis
- **THEN** the system classifies the feature as `face_milling` or `3_axis_milling`

#### Scenario: Feature requiring non-orthogonal tool access
- **WHEN** a feature is cut into a face that is neither coaxial with the part's principal rotational axis nor reachable along one of the three orthogonal machine axes (e.g. a slot or hole on an angled or off-axis face of an otherwise rotational part)
- **THEN** the system classifies the feature as `5_axis_milling`

### Requirement: Part-level process rollup
The system SHALL produce one part-level summary per part listing a primary process (the operation required by the largest share of the part's features, or the operation that defines the part's base body) and zero or more secondary processes, each with a rationale.

#### Scenario: All features share one operation
- **WHEN** every detected feature on a part classifies to the same operation
- **THEN** the part-level summary reports that operation as the sole primary process with an empty secondary-process list

#### Scenario: Turned body with a non-coaxial secondary feature
- **WHEN** a part's base body is axisymmetric (classified `turning`) but at least one feature (e.g. a keyway or off-axis hole) is not coaxial with the principal axis
- **THEN** the part-level summary reports `turning` as the primary process and reports the non-coaxial feature's operation as a secondary process, with a rationale explaining that the secondary feature cannot be produced by turning alone

### Requirement: Rationale reporting
The system SHALL report a human-readable rationale string alongside every feature-level and part-level classification, citing the geometric evidence used.

#### Scenario: Feature-level rationale
- **WHEN** a feature is classified
- **THEN** the output includes a rationale string that references the primitive type and axis relationship that drove the classification (e.g. "cylindrical face coaxial with part axis")

### Requirement: Unclassifiable feature handling
The system SHALL report `unknown` rather than guessing when a feature's underlying primitive type or geometric relationships are insufficient to determine an operation.

#### Scenario: Primitive type is unknown
- **WHEN** a feature's underlying primitive type is `unknown` (unclassified surface)
- **THEN** the system reports the feature's operation as `unknown` with a rationale noting insufficient geometric data, and excludes it from the primary-process vote

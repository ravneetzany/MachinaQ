## Purpose

Classify STEP B-Rep geometric primitives into named manufacturing features (hole, boss, slot, thread, drill) using primitive type plus dimensional and face-adjacency evidence, so each face receives at most one feature label with a stated rationale, instead of every face matching several unrelated rules at once.

## ADDED Requirements

### Requirement: Exclusive per-face classification
The system SHALL assign each geometric primitive (identified by `face_id`) to at most one feature type, chosen using more than primitive type alone (e.g. dimensions, proportions, or face-adjacency topology).

#### Scenario: Planar face with no supporting slot evidence
- **WHEN** a face is planar and has no dimensional or adjacency evidence indicating a slot (e.g. it is not a narrow, elongated pocket bounded by parallel walls)
- **THEN** the system does not classify it as a `slot`, and it is not spuriously labeled with any other feature type either

#### Scenario: Cylindrical face with no thread evidence
- **WHEN** a face is cylindrical but there is no evidence of a thread (e.g. no matching pitch/thread designation from the parser's standards data)
- **THEN** the system does not classify it as a `thread`

#### Scenario: A face receives exactly one label when qualifying evidence exists
- **WHEN** a face's dimensions and adjacency satisfy the evidence rule for exactly one feature type
- **THEN** the system's output contains exactly one `Feature` entry referencing that face's `face_id`, not multiple entries for the same face

### Requirement: Standards-validated hole detection reused, not re-derived
The system SHALL source `hole` features from the STEP parser's existing topology- and standards-based hole detection (through/blind classification, ASME/ISO size matching), rather than independently re-deriving holes from raw cylindrical primitives with weaker evidence.

#### Scenario: Hole feature carries through/blind and standard classification
- **WHEN** the parser has identified a standards-matched hole with a through/blind classification
- **THEN** the corresponding `hole` feature in the output includes that through/blind status and standard/label information

### Requirement: Full primitive coverage
The system SHALL make planar and conical primitives available to feature detection, in addition to cylindrical primitives, so rules that depend on non-cylindrical evidence can actually execute.

#### Scenario: Planar primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes planar surfaces
- **THEN** those planar primitives are present in the primitive list passed into feature detection (not silently dropped before detection runs)

#### Scenario: Conical primitives reach the detector
- **WHEN** a STEP file's parsed geometry includes conical surfaces
- **THEN** those conical primitives are present in the primitive list passed into feature detection

### Requirement: Rationale reporting
The system SHALL report a human-readable rationale for every detected feature, citing the geometric evidence that justified the classification.

#### Scenario: Feature includes rationale
- **WHEN** a feature is detected
- **THEN** its output includes a rationale string describing the evidence used (e.g. "cylindrical face matched ISO metric thread pitch" or "planar face bounded by two parallel long edges, width/length ratio below slot threshold")

### Requirement: Unclassified faces are reported, not dropped or guessed
The system SHALL report a face that matches no feature rule as unclassified rather than omitting it or assigning it a feature type without qualifying evidence.

#### Scenario: No rule matches
- **WHEN** a face's primitive type, dimensions, and adjacency do not satisfy any feature rule's evidence requirements
- **THEN** the system reports it as unclassified (not included in any `hole`/`boss`/`slot`/`thread`/`drill` list) rather than defaulting it into one of those categories

## ADDED Requirements

### Requirement: Large planar face detection
The system SHALL classify a planar primitive not already claimed by hole, thread, boss, or slot detection as a `planar_face` feature when it meets a size/extent threshold distinguishing a genuine large flat face (e.g. a stock or reference surface) from smaller or ambiguous planar surfaces, so it becomes eligible for operation classification rather than being reported as unclassified. This feature type is named `planar_face`, not `face`/`face_milling`, to stay distinct from the operation classifier's own `face_milling` operation label — the two are separate concepts (a detected geometric feature vs. a required-operation determination) and must not share a name.

#### Scenario: Large planar face is classified as a planar_face feature
- **WHEN** a planar face is not claimed by any hole, thread, boss, or slot rule, and its extent meets the large-face threshold
- **THEN** the system's output contains a `planar_face` feature entry referencing that face's `face_id`, with a rationale describing the extent evidence used

#### Scenario: Small or ambiguous planar face remains unclassified
- **WHEN** a planar face is not claimed by any hole, thread, boss, or slot rule, and its extent does not meet the large-face threshold
- **THEN** the system does not classify it as a `planar_face` feature, and it is reported as unclassified per the existing "Unclassified faces are reported, not dropped or guessed" requirement

#### Scenario: planar_face feature reaches operation classification
- **WHEN** a `planar_face` feature has been detected
- **THEN** it is included in the feature list passed to operation classification, so it receives a required-operation determination (e.g. 3-axis milling, 5-axis milling, or face milling) rather than being silently excluded

## MODIFIED Requirements

### Requirement: Unclassified faces are reported, not dropped or guessed
The system SHALL report a face that matches no feature rule — including the `planar_face` rule for large planar faces — as unclassified rather than omitting it or assigning it a feature type without qualifying evidence.

#### Scenario: No rule matches
- **WHEN** a face's primitive type, dimensions, and adjacency do not satisfy any feature rule's evidence requirements (including the large-planar-face extent threshold)
- **THEN** the system reports it as unclassified (not included in any `hole`/`boss`/`slot`/`thread`/`drill`/`planar_face` list) rather than defaulting it into one of those categories

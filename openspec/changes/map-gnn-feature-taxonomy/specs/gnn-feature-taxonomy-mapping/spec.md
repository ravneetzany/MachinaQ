## Purpose

Provide a documented, reasoned correspondence between the machining-feature taxonomies and representation approaches used by AAGNet, BRepNet, UV-Net, Hierarchical CADNet, and "Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning", and MachinaQ's own `Feature.feature_type`/operation-type schema, so a future integration change has an unambiguous mapping contract instead of re-deriving it ad hoc.

## ADDED Requirements

### Requirement: Per-reference classification
The mapping document SHALL, for each of the five referenced works, state whether that work defines its own machining-feature class taxonomy or is primarily a representation/message-passing technique evaluated against an existing external taxonomy or dataset, and SHALL cite the taxonomy or dataset it targets.

#### Scenario: Taxonomy-defining reference
- **WHEN** the document covers a reference that introduces its own fixed set of machining-feature classes (e.g. AAGNet's MFInstSeg classes, Hierarchical CADNet's MFCAD++ classes)
- **THEN** the document lists that reference's class set and names the dataset/paper it originates from

#### Scenario: Representation-technique reference
- **WHEN** the document covers a reference that is primarily an architecture or representation method rather than a taxonomy source (e.g. UV-Net's UV-grid representation, BRepNet's topological message passing)
- **THEN** the document states this explicitly and names the existing taxonomy/dataset that reference was evaluated against, instead of presenting it as if it defined a new independent class list

### Requirement: Class-level mapping table
For each taxonomy-defining reference, the mapping document SHALL provide a table mapping every source class to either a MachinaQ `Feature.feature_type` value or an explicit "unmapped" designation.

#### Scenario: Class maps to an existing MachinaQ feature type
- **WHEN** a source taxonomy class corresponds to a feature type MachinaQ's `Feature.feature_type` already represents (e.g. a through-hole class mapping to `hole`)
- **THEN** the table records that correspondence

#### Scenario: Class has no current MachinaQ equivalent
- **WHEN** a source taxonomy class (e.g. chamfer, pocket, step variants) does not correspond to any existing `Feature.feature_type` value
- **THEN** the table records it as `unmapped` rather than approximating it into an unrelated existing type

### Requirement: Gap summary
The mapping document SHALL include a consolidated list of every class marked `unmapped` across all taxonomy-defining references, deduplicated where classes are equivalent across references.

#### Scenario: Consolidated gap list present
- **WHEN** one or more source classes across the covered references are marked `unmapped`
- **THEN** the document includes a single summary section listing each distinct unmapped concept and which reference(s) it came from

### Requirement: Canonical taxonomy recommendation
The mapping document SHALL recommend one taxonomy (or a MachinaQ-specific superset of one) as the target class list for MachinaQ's future feature-schema evolution, with a stated rationale referencing the per-reference and gap analysis above.

#### Scenario: Recommendation is traceable to the analysis
- **WHEN** the document makes its canonical-taxonomy recommendation
- **THEN** the rationale references specific findings from the per-reference classification and gap summary sections (not an unsupported assertion)

### Requirement: No implied runtime change
The mapping document SHALL state explicitly that it does not require or imply any change to MachinaQ's current detection logic (`src/features.py`) or inference pipeline, and that adopting its recommendation is a decision for a future change.

#### Scenario: Scope boundary is explicit
- **WHEN** a reader consults the mapping document
- **THEN** it contains a explicit statement that the document is reference material only and that no code behavior changes as a result of this change

## Purpose

Train a learned CNC-operation classifier that supplements (never replaces) the existing rule-based `operation_classifier.py`, using labels self-distilled from that same rule engine over a multi-source geometry corpus, so a second, learned prediction path exists alongside the authoritative rules — the same relationship PointNet already has to `FeatureDetector`.

## ADDED Requirements

### Requirement: Rule-derived label generation
The system SHALL generate every training label by running the existing rule-based `operation_classifier.classify_feature()` (and `summarize_part()` for part-level labels) over each training example — not from any independently sourced ground truth, since none exists.

#### Scenario: Label matches what the rules would output today
- **WHEN** a training example is labeled
- **THEN** its label is exactly the `Operation` value `operation_classifier.classify_feature()` returns for that example's primitive/feature/principal-axis input, unmodified

### Requirement: Multi-source corpus assembly
The system SHALL assemble training examples from at least three sources, each via its existing ingestion path: NIST STEP files (`nist_sfa/*.stp`, `nist_sfa/holeTrain/*.step`, via `StepAnalyzer`), the OpenSCAD parts library (via `src/scad_ingest.py`), and the FreeCAD wrapper-script parts library (via `src/py_source_ingest.py`).

#### Scenario: STEP-sourced examples carry no principal axis
- **WHEN** a training example originates from a NIST STEP file
- **THEN** its `principal_axis` is `None` (STEP parsing does not compute one), matching the rule engine's own STEP-path behavior

#### Scenario: .scad/.py-sourced examples carry their ingester's principal axis
- **WHEN** a training example originates from the OpenSCAD or FreeCAD ingestion path
- **THEN** its `principal_axis` is whatever that ingester computed for the source part (a real `Axis`, or `None` for a part with no single rotational axis), matching the rule engine's own ingestion-path behavior

#### Scenario: A source that fails to parse contributes no examples
- **WHEN** a source file fails to ingest (STEP parse error, or a `.scad`/`.py` file reported unparsed by its ingester)
- **THEN** the assembly step skips that file and continues with the rest of the corpus, rather than aborting the whole assembly run

### Requirement: Feature vector representation
The system SHALL derive a fixed-size numeric feature vector per training example from the same signals `operation_classifier.classify_feature()` itself reads (primitive type, radius, axis-coaxiality with the principal axis or its absence, adjacent-planar-count, and bounding-extent ratio where available), not from a separately invented representation.

#### Scenario: Feature vector is deterministic for identical input
- **WHEN** the same primitive/feature/principal-axis input is vectorized twice
- **THEN** the resulting feature vector is identical both times

### Requirement: Trained model checkpoint
The system SHALL provide a training entry point that produces a saved model checkpoint, following the existing `run_train.py --model <type>` pattern used by the other trainable models in this repo.

#### Scenario: Checkpoint is produced by the standard training entry point
- **WHEN** `run_train.py --model operation-classifier` is run to completion against an available corpus
- **THEN** a checkpoint file is written to `outputs/`, in the same manner `--model unified`/`--model pointnet` already do

### Requirement: Supplementary inference, not a replacement
When a trained checkpoint is available, the system SHALL add its predictions to `StepAnalyzer.analyze()`'s report as an additional, clearly-labeled field, without altering the existing rule-derived `operation`/`operations_summary` fields' values or presence.

#### Scenario: Checkpoint present
- **WHEN** a trained operation-classifier checkpoint exists and `StepAnalyzer.analyze()` is run
- **THEN** the report includes both the existing rule-derived `operation`/`operations_summary` fields (unchanged) and an additional `operation_predictions` field carrying the model's own prediction and confidence per feature

#### Scenario: Checkpoint absent
- **WHEN** no trained operation-classifier checkpoint exists and `StepAnalyzer.analyze()` is run
- **THEN** the report is produced exactly as before this change (rule-derived fields only, no error), the same graceful-absence behavior the existing PointNet path already has

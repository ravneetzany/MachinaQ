## Why

`src/operation_classifier.py` (from the completed `add-cnc-operation-classifier` change) is a hand-written rule engine: it classifies each feature's required CNC operation from primitive type and axis-coaxiality alone. It has no learned component, so it can never generalize beyond the explicit rules its author wrote — every other MachinaQ prediction path (PointNet, PointNetBinary, MachinaQUnified) has a trained-model counterpart to its rule-based/heuristic sibling, but operation classification does not. There is also no human-labeled dataset of "correct CNC operation per feature" anywhere, so the only available training signal today is the rule engine's own output.

## What Changes

- Add a new trainable classifier (`models/operation_classifier_net.py`, a small feedforward network — not a point-cloud/graph model, since its input is the same small set of geometric signals the rule engine already uses: primitive type, radius, axis-coaxiality with the part's principal axis, adjacent-planar-count, and bounding-extent ratio) that predicts one of `operation_classifier.Operation`'s six labels (`turning`, `drilling`, `face_milling`, `3_axis_milling`, `5_axis_milling`, `unknown`).
- Add a dataset-assembly step that builds `(feature_vector, label)` training pairs by running the *existing* rule-based `operation_classifier.classify_feature()` over a corpus assembled from three already-working ingestion paths in this repo: NIST STEP files (`nist_sfa/*.stp`, `nist_sfa/holeTrain/*.step`, via `StepAnalyzer`), the OpenSCAD parts library (via `src/scad_ingest.py`), and the FreeCAD wrapper-script parts library (via `src/py_source_ingest.py`). **This makes the model self-distilled from the rules — it is not independent ground truth, and cannot become more "correct" than the rules that labeled it.** This is a deliberate, explicit choice (per the user), not an oversight.
- Add a training entry point, following the existing `run_train.py --model <type>` pattern: a new `--model operation-classifier` choice, backed by `src/train_operation_classifier.py` (mirroring `src/train_unified.py`'s structure), saving to `outputs/machinaq_operation_classifier.pth`.
- Wire the trained model into `StepAnalyzer.analyze()` as an *optional supplementary* prediction, the same way `PointNet` already supplements `FeatureDetector`: when a checkpoint exists, the report gains an `operation_predictions` list (model output + confidence) alongside the existing rule-derived `operation`/`operations_summary` fields. The rule-based classifier remains authoritative and unmodified; this change adds a second, learned opinion, not a replacement.
- **Not included**: AAGNet's own STEP corpus (the MFInstSeg dataset) — `aagnet/` is not cloned and the ~20 GB dataset is not downloaded in this environment, so it is excluded from the training corpus for now. Also not included: any change to `operation_classifier.py`'s rules or `Operation` values, and any change to `FeatureDetector`/`scad_ingest.py`/`py_source_ingest.py`'s existing behavior (this change only *consumes* their output).

## Capabilities

### New Capabilities
- `operation-classifier-training`: a learned CNC-operation classifier trained on rule-generated labels from a multi-source geometry corpus (STEP + `.scad` + FreeCAD), producing a checkpoint and an optional supplementary prediction path in the analysis report.

### Modified Capabilities
- (none — `operation_classifier.py`'s existing rule-based behavior, and the report fields it already populates, are unchanged; this change only adds new, additive fields/paths)

## Impact

- New: `models/operation_classifier_net.py`, `src/train_operation_classifier.py`, a dataset-assembly module (e.g. `src/operation_classifier_dataset.py`) that runs the ingestion + rule-labeling pipeline described above.
- Modified: `run_train.py` (new `--model` choice), `src/pipeline.py` (`StepAnalyzer` optionally loads the new checkpoint and adds `operation_predictions` to its report, mirroring the existing `self.model`/`predictions` PointNet pattern).
- No new third-party dependencies (reuses `torch`, already a project dependency).
- Data/compute: training runs against files already present in this repo (`nist_sfa/`) and the two external parts libraries at `/home/ravneetzany/projects/*-parts-library` already used by `add-cnc-operation-classifier`'s batch CLI — no new downloads required.

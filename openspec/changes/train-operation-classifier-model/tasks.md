## 1. Dataset assembly

- [x] 1.1 Create `src/operation_classifier_dataset.py` with a `TrainingExample` dataclass (feature vector inputs + label) and a `vectorize(primitive, principal_axis) -> Tuple[float, ...]` function implementing design.md decision 1's 8-dim encoding; verify with a unit test asserting the same input always produces the same vector (determinism)
- [x] 1.2 Implement `build_step_examples(step_paths) -> List[TrainingExample]`: run each STEP file through `StepAnalyzer`-equivalent parsing (`StepTextParser` + `_extract_primitives`-equivalent + `FeatureDetector`), label each feature via `operation_classifier.classify_feature(..., principal_axis=None)`, and skip (not raise on) files that fail to parse; verify against `nist_sfa/holeTrain/*.step` and at least one `nist_sfa/*.stp` file, asserting examples are produced and a deliberately-broken/missing path is skipped without raising
- [x] 1.3 Implement `build_scad_examples(library_root) -> List[TrainingExample]` using `scad_ingest.ingest_directory()`, labeling each ingested primitive as its own feature (per the established `add-cnc-operation-classifier` pattern) via `classify_feature(..., principal_axis=result.principal_axis)`; verify against `/home/ravneetzany/projects/openscad-parts-library`, asserting examples are produced and unparsed files contribute none (no exception)
- [x] 1.4 Implement `build_freecad_examples(library_root) -> List[TrainingExample]` using `py_source_ingest.ingest_directory()`, same labeling approach; verify against `/home/ravneetzany/projects/freecad-parts-library`, asserting examples are produced with no exception
- [x] 1.5 Implement `build_corpus(nist_paths, scad_root, freecad_root) -> List[TrainingExample]` combining 1.2-1.4, logging per-source and per-class example counts (per design.md decision/risk section); verify by running it end-to-end and asserting the combined list is non-empty and covers examples from all three sources

## 2. Model

- [x] 2.1 Create `models/operation_classifier_net.py` with `OperationClassifierNet(nn.Module)` (8 → 32 → 16 → 6 MLP per design.md decision 3) and a `load_model()` helper matching `models/pointnet.py`'s existing signature; verify by instantiating the model, running a forward pass on a random `(4, 8)` tensor, and asserting the output shape is `(4, 6)`

## 3. Training entry point

- [x] 3.1 Create `src/train_operation_classifier.py` with a `train_operation_classifier(epochs=..., batch_size=..., lr=...)` function following `train_unified.py`'s structure (build dataset via `operation_classifier_dataset.build_corpus`, train/val split, per-epoch logging, save best checkpoint to `outputs/machinaq_operation_classifier.pth`); verify by running a short training run (a few epochs) end-to-end against the real corpus and confirming a checkpoint file is written
- [x] 3.2 Add `operation-classifier` to `run_train.py`'s `--model` choices, wired to call `train_operation_classifier`, following the existing branch pattern for `unified`/`pointnet`; verify by running `python run_train.py --model operation-classifier --epochs 2` and confirming it completes and logs to `outputs/operation_classifier_train.log`

## 4. Inference wiring

- [x] 4.1 In `StepAnalyzer.__init__`/a new `_load_operation_model()`, optionally load `outputs/machinaq_operation_classifier.pth` into `self.operation_model`, guarded the same try/except-and-warn way `_load_model()` guards PointNet's absence; verify with a unit test asserting `StepAnalyzer()` instantiates without error whether or not the checkpoint file exists
- [x] 4.2 In `StepAnalyzer.analyze()`, when `self.operation_model is not None`, compute and add an `operation_predictions` list (per-feature model prediction + confidence, using the same `vectorize()` from task 1.1) to the returned report, without altering the existing `operation`/`operations_summary` fields; verify with a unit test asserting: (a) with no checkpoint present, the report has no `operation_predictions` key and all existing fields are unchanged from before this change; (b) with a checkpoint present (train one first, or use a freshly-initialized model for the test), the report includes `operation_predictions` alongside unchanged existing fields

## 5. Tests and docs

- [x] 5.1 Add unit tests under `tests/` for `operation_classifier_dataset.py` and `operation_classifier_net.py` covering every scenario in `specs/operation-classifier-training/spec.md`; verify with `pytest tests/` passing
- [x] 5.2 Update `README.md`'s Training section (model types table) to add the `operation-classifier` entry, and note in its `/analyze` endpoint description that `operation_predictions` appears when a checkpoint is present; verify by re-reading the updated sections for accuracy against the shipped module/flag names

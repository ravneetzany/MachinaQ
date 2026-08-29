## Context

See proposal.md - Why. Relevant current state:

- `src/operation_classifier.py`'s `Operation` has six values (`turning`, `drilling`, `face_milling`, `3_axis_milling`, `5_axis_milling`, `unknown`). `classify_feature(feature, primitives_by_face, principal_axis)` reads only: the referenced primitive(s)' `type`, `details['radius']`, axis data recoverable via `geometry.axis_from_details`, and whether `principal_axis` is `None`.
- Three ingestion paths already exist and produce `SurfacePrimitive` + (for `.scad`/`.py`) a `principal_axis`: `StepAnalyzer._extract_primitives()` (STEP, `principal_axis` always `None`), `scad_ingest.ingest_file()`/`ingest_directory()`, `py_source_ingest.ingest_file()`/`ingest_directory()`.
- `run_train.py` already has a uniform CLI pattern (`--model <type>`, shared `--epochs`/`--batch-size`/`--lr`, per-model log file, save to `outputs/machinaq_<type>.pth`) that `src/train_unified.py`/`train_through_hole.py` etc. implement against.
- `StepAnalyzer.analyze()` already has an established "optional supplementary model" pattern: `self.model` is loaded if `models/pointnet_trained.pth` exists (`_load_model()`), and `predictions` in the report is `[]` when it doesn't — the new checkpoint follows the same shape.
- `aagnet/` is an empty directory in this environment (never cloned) and the MFInstSeg dataset (~20 GB) is not downloaded — per the user's explicit scoping, these are excluded from this change's corpus.

## Goals / Non-Goals

**Goals:**
- A feedforward classifier trained on rule-generated labels from STEP + `.scad` + FreeCAD sources.
- A `run_train.py --model operation-classifier` entry point producing a checkpoint the same way other models do.
- An optional, clearly-separate `operation_predictions` field in `StepAnalyzer.analyze()`'s report when a checkpoint exists — never altering the existing rule-derived fields.

**Non-Goals:**
- No AAGNet/MFInstSeg data (not available in this environment; a future change can add it once the dataset exists locally).
- No change to `operation_classifier.py`'s rules, `Operation` values, or their behavior — this change only reads their output as training labels.
- No change to `FeatureDetector`, `scad_ingest.py`, or `py_source_ingest.py`'s existing behavior — this change only consumes what they already produce.
- No claim of independent accuracy: since labels come entirely from the rule engine, evaluation only measures how well the model reproduces the rules on held-out examples, not real-world correctness. This limitation is stated in the training script's output, not hidden.
- No point-cloud/graph model (unlike PointNet/AAGNet) — the classifier's input is a small fixed-size vector of the same signals the rules already use, so a small MLP is the appropriate architecture, not a heavier point-cloud encoder.

## Decisions

**1. Feature vector: a fixed 8-dimensional encoding of exactly what `classify_feature()` reads.**
`[is_cylindrical, is_conical, is_planar, is_unknown_type, log1p(radius or 0), has_principal_axis, is_coaxial_with_principal_axis (0 if no axis), is_axis_aligned_with_orthogonal (0 if no axis)]`. Rationale: the rule engine's entire decision surface is primitive type (one-hot, 4 values) + radius (log-scaled since raw mm values span orders of magnitude across the corpus, from small drilled holes to whole part envelopes) + two axis-relationship booleans it already computes internally (`is_coaxial`, `is_axis_aligned_with_any` from `geometry.py`) + whether an axis exists at all. Reusing these exact signals means the model is learning the same feature space the rules use, not a richer or different one — appropriate for a self-distillation setup where the goal is a light-weight learned approximation, not new information the rules don't have. Alternative considered: feed raw axis vectors/points directly — rejected, since the rules never use raw coordinates, only the derived coaxiality/alignment booleans, and feeding more would let the model key off details irrelevant to the actual rule logic.

**2. Dataset assembly is a distinct module (`src/operation_classifier_dataset.py`), not inlined in the training script.**
`build_examples(corpus_roots) -> List[TrainingExample]` walks all three sources, calls each one's existing ingestion path, runs `FeatureDetector`-equivalent feature construction per source (reusing `add-cnc-operation-classifier`'s established "each `.scad`/`.py` primitive is its own feature" pattern for those two sources, and the full STEP `FeatureDetector`+parser-hole path for NIST files), and labels each resulting feature via `operation_classifier.classify_feature()`. Kept separate from `src/train_operation_classifier.py` so the corpus-building logic (which sources, which ingestion path per source, how failures are skipped) is unit-testable independent of the training loop, mirroring how `UnifiedHoleDataset` in `train_unified.py` is itself a `torch.utils.data.Dataset` built from a clearly separable data-generation step.

**3. Model architecture: a 3-layer MLP (8 → 32 → 16 → 6), following `PointNet`'s existing "small `nn.Module` in `models/`" convention.**
`models/operation_classifier_net.py` defines `OperationClassifierNet(nn.Module)` with a `forward(x: (B, 8)) -> (B, 6)` logits output over `Operation.ALL`'s six values, plus a `load_model`/save-path helper matching `models/pointnet.py`'s existing `load_model()` signature so `train_operation_classifier.py` and `pipeline.py` can load it the same way `PointNet` is loaded today.

**4. Training script mirrors `train_unified.py`'s structure exactly: build dataset → split → train loop with per-epoch logging → save best checkpoint.**
`src/train_operation_classifier.py` exposes a `train_operation_classifier(epochs=..., batch_size=..., lr=...)` function `run_train.py` calls for the new `--model operation-classifier` branch, logging to `outputs/operation_classifier_train.log` and saving to `outputs/machinaq_operation_classifier.pth`, matching every other model's naming convention.

**5. Inference wiring in `pipeline.py` mirrors the existing PointNet path exactly, added alongside it (not replacing it).**
`StepAnalyzer.__init__` gains a second optional load (`self.operation_model`, guarded the same try/except-and-warn way `_load_model()` already guards PointNet's absence), and `analyze()` gains an `operation_predictions` list in its returned dict — computed only when `self.operation_model is not None`, using the same 8-dim vectorization from decision 1 applied to each feature already computed for the rule-based `operation`/`operations_summary` fields. The existing fields are built exactly as they are today; this is a pure addition.

## Risks / Trade-offs

- **[Risk]** Self-distillation means the model can only ever match or underperform the rules it was trained on — it adds no new correctness, only a different (learned, potentially faster or more tolerant-of-noisy-input) way of approximating the same decision boundary. → **Mitigation**: this is the explicit, user-confirmed scope (see proposal.md), documented in the training script's own output/logs so a reader of `outputs/operation_classifier_train.log` sees this stated, not just implied.
- **[Risk]** The corpus is small (5 NIST holeTrain files + a handful of NIST STEP files + ~10 `.scad`/`.py` parts) and heavily class-imbalanced (per `add-cnc-operation-classifier`'s own spot-check, most STEP-path features currently land on `drilling`/`3_axis_milling` under the no-axis fallback) — a model trained on this may trivially predict the majority class. → **Mitigation**: `train_operation_classifier.py` logs per-class example counts before training (matching `train_unified.py`'s existing dataset-summary logging pattern) so this imbalance is visible, not hidden; not a blocker for this change, since the explicit goal is a working supplementary path, not a benchmarked model.
- **[Trade-off]** Excluding AAGNet/MFInstSeg (Non-Goals) means the corpus has no B-Rep-GNN-labeled examples — acceptable since that data isn't available in this environment, and the multi-source (STEP + `.scad` + FreeCAD) corpus already exercises both the axis-aware and axis-absent branches of the rule engine.

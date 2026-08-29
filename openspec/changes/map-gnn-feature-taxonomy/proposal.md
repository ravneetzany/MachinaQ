## Why

MachinaQ already has an AAGNet training subproject (`train_aagnet.py`, `setup_aagnet.bat` cloning `github.com/whjdark/AAGNet`, targeting the MFInstSeg per-face taxonomy) but no trained checkpoint yet, and AAGNet's output has never been mapped onto MachinaQ's own feature representation (`src/features.py`'s `Feature.feature_type`: hole/boss/slot/thread/drill, soon revised by the pending `improve-feature-detection-rules` change) or onto the pending `add-cnc-operation-classifier` change's operation types. Before wiring any GNN model into inference, MachinaQ needs a documented, reasoned mapping from B-Rep-GNN machining-feature taxonomies — AAGNet plus BRepNet, UV-Net, Hierarchical CADNet, and "Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning" — onto its own schema, so a future integration change has an unambiguous contract instead of guessing class correspondences ad hoc.

## What Changes

- Add a documented **feature-taxonomy mapping** capability: a reference document (and, if a stable structured form is warranted, a machine-readable table) that, for each of the five referenced works, records the machining-feature classes it recognizes and how each maps onto MachinaQ's `Feature.feature_type` values, flagging classes with no current equivalent (e.g. chamfer, pocket, step variants) as schema gaps rather than silently dropping or misassigning them.
- Record, for each reference, what it actually contributes to this mapping (a fixed class taxonomy vs. a representation/architecture technique) — UV-Net and BRepNet are primarily representation/message-passing approaches evaluated on existing taxonomies (e.g. MFCAD-family datasets) rather than sources of a *new* taxonomy, so the proposal's "map onto MachinaQ's schema" applies at different depths per reference; the mapping document states this explicitly per reference instead of forcing every reference into an identical taxonomy table.
- Recommend (in the design) which taxonomy MachinaQ should adopt as its canonical target class list going forward, and how `Feature.feature_type` should evolve to accommodate it, without implementing that schema change in this pass.
- **Not included**: no model training, no inference wiring into `src/pipeline.py`/`src/api.py`, no changes to `src/features.py`'s actual detection rules (that's `improve-feature-detection-rules`'s scope) — this change produces the mapping/reference artifact only, per the user's explicit choice to scope this narrowly.

## Capabilities

### New Capabilities
- `gnn-feature-taxonomy-mapping`: a documented, reasoned correspondence between the machining-feature taxonomies used by AAGNet, BRepNet, UV-Net, Hierarchical CADNet, and the graph-representation-for-MFR paper, and MachinaQ's own `Feature`/operation-type schema, including explicit gap flags for unmapped classes.

### Modified Capabilities
- (none — this change only adds a reference/mapping artifact; it does not change any existing runtime behavior or spec)

## Impact

- New: a reference document under `docs/` (e.g. `docs/GNN_FEATURE_TAXONOMY_MAPPING.md`) and, if warranted per design, a structured sidecar (e.g. `docs/feature_taxonomy_mapping.json` or `.yaml`) that a future integration change can load rather than re-deriving.
- No code in `src/`, `models/`, or `aagnet/` is modified by this change.
- Downstream: this mapping is a direct input to any future change that wires AAGNet (or another of these references) into inference, and to `add-cnc-operation-classifier`'s eventual extension to GNN-derived features.

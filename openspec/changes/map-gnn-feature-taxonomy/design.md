## Context

See proposal.md - Why. Relevant current state:

- MachinaQ's own taxonomy today is `Feature.feature_type` in `src/features.py`: `hole`, `boss`, `slot`, `thread`, `drill` (five values, and the pending `improve-feature-detection-rules` change tightens their detection rules but does not add new class names).
- `train_aagnet.py` already targets AAGNet's MFInstSeg dataset/taxonomy (per its docstring: "24 classes: slot, hole, chamfer, pocket …" plus instance and bottom-face segmentation) but nothing in `src/` consumes AAGNet output yet.
- The other four references are not present in this repo at all today — this change's research work (reading each paper/repo's stated taxonomy) is the first time MachinaQ records what they classify.
- This is a documentation-only change: the "system" whose behavior the spec constrains is the mapping document itself (its required sections/content), not runtime code.

## Goals / Non-Goals

**Goals:**
- Produce one authoritative reference document classifying all five works as taxonomy-defining or representation-technique references, with citations.
- Produce class-level mapping tables (source class → MachinaQ `feature_type` or `unmapped`) for each taxonomy-defining reference.
- Produce a consolidated gap list and a canonical-taxonomy recommendation with rationale.

**Non-Goals:**
- No training run, no checkpoint, no inference wiring (explicitly deferred per the user's scoping choice).
- No change to `src/features.py`'s `Feature.feature_type` enum/values themselves — the recommendation is documented, not implemented, in this change.
- Does not attempt to fully catalog every architectural detail of each paper (e.g. UV-Net's exact CNN/GNN layer structure) beyond what's needed to state which taxonomy/dataset it was evaluated against — deep architecture summaries are out of scope for a taxonomy-mapping document.

## Decisions

**1. One Markdown document, `docs/GNN_FEATURE_TAXONOMY_MAPPING.md`, is the primary artifact; no separate machine-readable sidecar in this pass.**
Rationale: the spec's requirements (per-reference classification, class tables, gap summary, recommendation, scope-boundary statement) are all naturally satisfied by one well-structured document with tables; a future integration change that needs the mapping programmatically can parse the Markdown tables or generate a sidecar then, once the actual consuming code shape is known — inventing a JSON/YAML schema now, before any consumer exists, risks guessing wrong about what fields that consumer needs. Alternative considered: a structured YAML table now — rejected as premature structure for a document with no current programmatic reader.

**2. Per-reference depth follows what each reference actually is, not a forced-uniform template.**
- **AAGNet**: taxonomy-defining (MFInstSeg, ~24-25 per-face classes incl. instance/bottom-face segmentation) — full class table.
- **Hierarchical CADNet**: taxonomy-defining (targets MFCAD++, the dataset that paper itself introduced) — full class table.
- **"Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning"**: taxonomy-defining at the level of the machining-feature classes it classifies (an earlier graph+GNN approach, precursor in spirit to AAGNet's graph representation) — full class table.
- **BRepNet**: primarily a topological message-passing architecture (operates directly on B-Rep coedges/wingedges rather than a generic face-adjacency graph) evaluated against an existing per-face segmentation taxonomy (the MFCAD-family dataset) — documented as a representation technique, with a note on which existing taxonomy its published results target, not a new independent class table.
- **UV-Net**: primarily a representation-learning approach (UV-sampled grids per face/edge, joint CNN+GNN encoder) used for multiple downstream tasks (solid classification, segmentation) across several datasets (e.g. SolidLetters, Fusion 360 Gallery, MFCAD) — documented as a representation technique, not a taxonomy source; note which segmentation taxonomy its face-segmentation results (if any) target.
Rationale: matches the spec's explicit "Taxonomy-defining reference" vs. "Representation-technique reference" scenarios — forcing BRepNet/UV-Net into a fabricated independent class list would misrepresent what they actually contribute.

**3. Mapping granularity: source class → MachinaQ `feature_type`, not source class → CNC operation.**
The pending `add-cnc-operation-classifier` change already defines feature → operation rules on top of `feature_type`; re-deriving a second independent source-class → operation mapping here would create two divergent paths to the same answer. This document stops at `feature_type` (or `unmapped`), and operation classification composes on top of it via the existing/pending classifier, once a class is actually mapped into MachinaQ's schema.

**4. Gap classes are named descriptively, not force-fit into new invented `feature_type` values.**
E.g. "chamfer", "pocket (rectangular/triangular/six-sided)", "step (through/blind/slant)" are recorded as named gap concepts in the summary, not as newly-invented `feature_type` strings — inventing new enum values is a schema change, which is explicitly out of scope (Non-Goals) and belongs to whichever future change acts on the canonical-taxonomy recommendation.

**5. Recommendation approach: adopt the AAGNet/MFInstSeg class list as MachinaQ's long-term canonical target, documented as a recommendation only.**
Rationale to record in the document: it is the taxonomy MachinaQ already has training infrastructure pointed at (`train_aagnet.py`), it's a strict superset of the classes the other taxonomy-defining references cover (per this change's own comparison table), and it includes instance + bottom-face segmentation info the others don't. This is a recommendation for the document to state and justify — not something this change implements.

## Risks / Trade-offs

- **[Risk]** Class taxonomies and exact counts (e.g. "24 vs 25 classes") vary across MFCAD/MFCAD++/MFInstSeg paper versions and even the AAGNet repo's own README vs. code; a mapping table written from memory/secondary description risks being subtly wrong. → **Mitigation**: the tasks below require pulling each class list from the primary source (the referenced repo's dataset definition / the paper's own table), not from paraphrase, and citing exactly where each list came from.
- **[Risk]** Two of the five references (BRepNet, UV-Net) don't define an independent taxonomy, so a reader skimming only the mapping-table sections might expect five parallel tables and be confused by their absence. → **Mitigation**: spec's "Per-reference classification" requirement forces an explicit up-front statement of which category each reference falls into, before any tables appear.
- **[Trade-off]** Choosing Markdown-only (decision 1) over a structured sidecar means a future integration change must parse or hand-transcribe the table rather than load a data file — acceptable given no consumer exists yet; revisit once one does.

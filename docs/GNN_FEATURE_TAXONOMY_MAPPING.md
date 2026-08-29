# GNN Feature Taxonomy Mapping

**Scope boundary:** This document is reference material only. It does not require or imply any change to MachinaQ's current detection logic (`src/features.py`) or inference pipeline. Adopting its recommendation (see [Canonical Taxonomy Recommendation](#canonical-taxonomy-recommendation)) is a decision for a future change, not something this document implements.

## Purpose

MachinaQ's own machining-feature vocabulary today is `Feature.feature_type` in `src/features.py`: `hole`, `boss`, `slot`, `thread`, `drill`. Before any future change wires a B-Rep GNN model (AAGNet or otherwise) into inference, this document records — for five relevant works (AAGNet, BRepNet, UV-Net, Hierarchical CADNet, and "Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning") — what machining-feature taxonomy (if any) each one uses, and how each source class maps onto MachinaQ's own schema. Every class list below was pulled from the primary source (the referenced repository's own dataset-definition file, or the paper's own text), not from secondary paraphrase, and is cited to the exact file/section it came from.

## Per-Reference Classification

| Reference | Category | Taxonomy / dataset it targets |
|---|---|---|
| **AAGNet** (Wang et al., whjdark/AAGNet) | Taxonomy-defining | MFInstSeg — 24 machining-feature classes + 1 stock/background class |
| **Hierarchical CADNet** (Colligan et al., *Computer-Aided Design* 2022) | Taxonomy-defining | MFCAD++ — 24 machining-feature classes + 1 stock class (the dataset the paper itself introduced) |
| **"Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning"** (Cao, Robinson, Hua, Boussuge, Colligan, Pan; ASME IDETC-CIE 2020) | Taxonomy-defining | MFCAD — 15 machining-feature classes + 1 stock class |
| **BRepNet** (Lambourne et al., CVPR 2021) | Representation technique | Its own Fusion 360 Gallery segmentation dataset — 8 modeling-*operation* classes, not machining-feature classes |
| **UV-Net** (Jayaraman et al., CVPR 2021) | Representation technique | Multiple datasets; for machining-feature segmentation specifically, the same MFCAD taxonomy as the Graph Representation paper above |

AAGNet, Hierarchical CADNet, and the Graph Representation paper each introduce or target their own fixed machining-feature class list, so each gets a full class-mapping table below. BRepNet and UV-Net are primarily architecture/representation techniques evaluated across one or more existing datasets — they are documented as such, not forced into a fabricated independent taxonomy.

---

## Taxonomy-Defining References

### AAGNet — MFInstSeg (24 classes + stock)

**Source:** `whjdark/AAGNet`, file `dataset/Utils/parameters.py`, the `feat_names` list (lines 28–52), fetched directly from the repository. This is the definitive, code-level class-index list AAGNet trains against — not the repo's README (which describes the tasks but does not enumerate the classes) or MachinaQ's own `train_aagnet.py` docstring paraphrase ("24 classes: slot, hole, chamfer, pocket …").

**Cross-check against `train_aagnet.py`'s docstring:** the docstring's "24 classes" claim is confirmed accurate — `feat_names` has 24 machining-feature entries (indices 0–23) plus one additional `'stock'` background class (index 24), for 25 labels total. The docstring's illustrative examples ("slot, hole, chamfer, pocket") are all present in the real list.

AAGNet also performs **instance segmentation** (grouping faces into individual feature instances, not just per-face class labels) and **bottom-face segmentation** (identifying which faces are a feature's bottom/floor face) simultaneously with semantic segmentation — capabilities beyond a flat per-face class list, per the repository's stated multi-task design.

**Notable finding:** the same file also defines a second list, `feat_names_planar` (16 classes + stock), which is *identical in names and order* to the Graph Representation paper's MFCAD taxonomy (see below) — meaning AAGNet's own codebase already embeds both taxonomies (MFCAD's 16-class planar-only subset, and MFCAD++'s full 24-class set).

| # | AAGNet class | MachinaQ `feature_type` |
|---|---|---|
| 0 | chamfer | `unmapped` |
| 1 | through_hole | `hole` |
| 2 | triangular_passage | `unmapped` |
| 3 | rectangular_passage | `unmapped` |
| 4 | 6sides_passage | `unmapped` |
| 5 | triangular_through_slot | `slot` |
| 6 | rectangular_through_slot | `slot` |
| 7 | circular_through_slot | `slot` |
| 8 | rectangular_through_step | `unmapped` |
| 9 | 2sides_through_step | `unmapped` |
| 10 | slanted_through_step | `unmapped` |
| 11 | Oring | `unmapped` |
| 12 | blind_hole | `hole` |
| 13 | triangular_pocket | `unmapped` |
| 14 | rectangular_pocket | `unmapped` |
| 15 | 6sides_pocket | `unmapped` |
| 16 | circular_end_pocket | `unmapped` |
| 17 | rectangular_blind_slot | `slot` |
| 18 | v_circular_end_blind_slot | `slot` |
| 19 | h_circular_end_blind_slot | `slot` |
| 20 | triangular_blind_step | `unmapped` |
| 21 | circular_blind_step | `unmapped` |
| 22 | rectangular_blind_step | `unmapped` |
| 23 | round | `unmapped` |
| 24 | stock | `unmapped` (not a machining feature — background/blank material) |

**Summary:** 8 of 25 classes map onto MachinaQ's schema (2 → `hole`, 6 → `slot`); 17 are unmapped. None of AAGNet's classes correspond to MachinaQ's `boss`, `thread`, or `drill` types — MFInstSeg's taxonomy is entirely subtractive-cavity features (holes, slots, pockets, steps, passages) plus chamfer/round/stock; it has no protrusion (`boss`) or conical-feature (`drill`) concept at all.

### Hierarchical CADNet — MFCAD++ (24 classes + stock)

**Source:** `gitlab.com/qub_femg/machine-learning/mfcad2-dataset`, file `dataset_description.txt`, fetched directly from the repository (the official dataset repository linked from the paper).

```
0 Chamfer                      13 Triangular pocket
1 Through hole                 14 Rectangular pocket
2 Triangular passage           15 6-sides pocket
3 Rectangular passage          16 Circular end pocket
4 6-sides passage              17 Rectangular blind slot
5 Triangular through slot      18 Vertical circular end blind slot
6 Rectangular through slot     19 Horizontal circular end blind slot
7 Circular through slot        20 Triangular blind step
8 Rectangular through step     21 Circular blind step
9 2-sides through step         22 Rectangular blind step
10 Slanted through step        23 Round
11 O-ring                      24 Stock
12 Blind hole
```

This list is **identical in class names, order, and count** to AAGNet's `feat_names` above (differing only in capitalization/spacing) — Hierarchical CADNet's MFCAD++ dataset is the taxonomy AAGNet's MFInstSeg dataset extends. The class-mapping table is therefore the same as AAGNet's:

| # | MFCAD++ class | MachinaQ `feature_type` |
|---|---|---|
| 0 | Chamfer | `unmapped` |
| 1 | Through hole | `hole` |
| 2 | Triangular passage | `unmapped` |
| 3 | Rectangular passage | `unmapped` |
| 4 | 6-sides passage | `unmapped` |
| 5 | Triangular through slot | `slot` |
| 6 | Rectangular through slot | `slot` |
| 7 | Circular through slot | `slot` |
| 8 | Rectangular through step | `unmapped` |
| 9 | 2-sides through step | `unmapped` |
| 10 | Slanted through step | `unmapped` |
| 11 | O-ring | `unmapped` |
| 12 | Blind hole | `hole` |
| 13 | Triangular pocket | `unmapped` |
| 14 | Rectangular pocket | `unmapped` |
| 15 | 6-sides pocket | `unmapped` |
| 16 | Circular end pocket | `unmapped` |
| 17 | Rectangular blind slot | `slot` |
| 18 | Vertical circular end blind slot | `slot` |
| 19 | Horizontal circular end blind slot | `slot` |
| 20 | Triangular blind step | `unmapped` |
| 21 | Circular blind step | `unmapped` |
| 22 | Rectangular blind step | `unmapped` |
| 23 | Round | `unmapped` |
| 24 | Stock | `unmapped` (background) |

**Summary:** same as AAGNet — 8 of 25 mapped (2 `hole`, 6 `slot`), 17 unmapped, none map to `boss`/`thread`/`drill`.

### "Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning" — MFCAD (15 classes + stock)

**Source:** Cao, Robinson, Hua, Boussuge, Colligan, Pan, ASME IDETC-CIE 2020. Code repository `gitlab.com/qub_femg/machine-learning/cadnet` (linked from the paper's official `AndrewColligan/CADNet` GitHub mirror), file `mfcad_dataset_description.txt`, fetched directly.

```
0  Rectangular through slot     8  Rectangular blind step
1  Triangular through slot      9  Triangular blind step
2  Rectangular passage         10  Rectangular blind slot
3  Triangular passage          11  Rectangular pocket
4  6 sided passage             12  Triangular pocket
5  Rectangular through step    13  6 sided pocket
6  2 sided through step        14  Chamfer
7  Slanted through step        15  Stock
```

| # | MFCAD class | MachinaQ `feature_type` |
|---|---|---|
| 0 | Rectangular through slot | `slot` |
| 1 | Triangular through slot | `slot` |
| 2 | Rectangular passage | `unmapped` |
| 3 | Triangular passage | `unmapped` |
| 4 | 6 sided passage | `unmapped` |
| 5 | Rectangular through step | `unmapped` |
| 6 | 2 sided through step | `unmapped` |
| 7 | Slanted through step | `unmapped` |
| 8 | Rectangular blind step | `unmapped` |
| 9 | Triangular blind step | `unmapped` |
| 10 | Rectangular blind slot | `slot` |
| 11 | Rectangular pocket | `unmapped` |
| 12 | Triangular pocket | `unmapped` |
| 13 | 6 sided pocket | `unmapped` |
| 14 | Chamfer | `unmapped` |
| 15 | Stock | `unmapped` (background) |

**Summary:** 3 of 16 classes map onto MachinaQ's schema (all 3 → `slot`); 13 are unmapped. MFCAD has **no hole classes at all** — it is a purely planar/milling-feature dataset (consistent with AAGNet's own `feat_names_planar` subset, found above, being this exact 16-class list). None of MFCAD's classes map to `boss`, `thread`, or `drill` either.

---

## Representation-Technique References

### BRepNet

BRepNet is **not** a taxonomy source — it is a topological message-passing architecture that defines convolutional kernels directly over B-Rep coedges (rather than a generic face-adjacency graph), so it can operate on the exact B-Rep data structure without approximating it as a mesh or point cloud.

**What it was actually evaluated on:** the paper (Lambourne et al., CVPR 2021) introduces and evaluates exclusively on its own **Fusion 360 Gallery segmentation dataset** — 35,858 B-Rep models with per-face labels drawn from **8 modeling-*operation* classes** (`ExtrudeSide`, `ExtrudeEnd`, `CutSide`, `CutEnd`, `Fillet`, `Chamfer`, `RevolveSide`, `RevolveEnd`), reporting 92.52% ± 0.15% accuracy / 77.10% ± 0.54% IoU. Source: the paper's arXiv HTML rendering (`arxiv.org/html/2104.00706v2`), experiments section.

**Correction from an earlier assumption:** this change's design initially assumed BRepNet was evaluated against the MFCAD-family machining-feature taxonomy. Verifying against the actual paper text shows this is incorrect — MFCAD (Cao et al.) is cited only in BRepNet's Related Work section as prior graph-convolution work, and is never used in BRepNet's own experiments. BRepNet's 8 classes label *what modeling operation created a face* (a CAD-history concept), not *what machining feature a face belongs to* — a materially different labeling task from the taxonomy-defining references above. No class-mapping table is produced for BRepNet; its 8 operation labels don't correspond to MachinaQ's feature vocabulary in any direct way, and mapping "which operation created this face" onto "what machining feature is this" would require a modeling-history signal MachinaQ's B-Rep/STEP pipeline doesn't have.

### UV-Net

UV-Net is also **not** a taxonomy source — it is a representation-learning approach that encodes each B-Rep face/edge via a UV-sampled parametric grid (processed with a CNN) combined with the topology adjacency graph (processed with a GNN), used as a general-purpose B-Rep encoder for multiple downstream tasks.

**What it was evaluated on** (Jayaraman et al., CVPR 2021, verified via `ar5iv.labs.arxiv.org/html/2006.10211`):

| Dataset | Task | Classes |
|---|---|---|
| Machining Feature Dataset (its own) | whole-solid classification | 24 |
| FabWave | whole-solid classification | 52 mechanical part categories |
| SolidLetters | whole-solid classification | 26 (alphabet letters) |
| **MFCAD** (Cao et al.) | **per-face segmentation** | **16** |
| ABC (subset) | per-face segmentation | 6 modeling-operation classes |

For the one evaluation relevant to machining-feature *segmentation* specifically, UV-Net uses the **same 16-class MFCAD taxonomy** as the Graph Representation paper above — the paper states directly: *"MFCAD dataset [6] a synthetic segmentation dataset of 15,488 3D shapes, similar to the Machining feature dataset, but with multiple machining features. 16 different segmentation labels are used and applied per face,"* citing Cao et al. as reference [6]. Because this is the identical taxonomy already tabled under the Graph Representation paper's section above, no separate class table is produced for UV-Net — see that table. UV-Net's other four datasets (its own whole-solid Machining Feature Dataset, FabWave, SolidLetters, ABC) are whole-part classification or modeling-operation segmentation, not per-face machining-feature taxonomies, and are out of scope for this mapping.

---

## Gap Summary

Consolidated, deduplicated list of every `unmapped` concept found across the taxonomy-defining references (AAGNet, Hierarchical CADNet/MFCAD++, MFCAD), with which reference(s) each came from:

| Gap concept | Found in |
|---|---|
| **Chamfer** | AAGNet, MFCAD++, MFCAD |
| **Passage** (triangular / rectangular / 6-sided — a through-cavity of a given cross-section shape) | AAGNet, MFCAD++, MFCAD |
| **Step** (through: rectangular / 2-sided / slanted; blind: rectangular / triangular / circular — a change in surface level cutting across the part) | AAGNet, MFCAD++, MFCAD |
| **Pocket** (triangular / rectangular / 6-sided / circular-end — a closed-bottom cavity) | AAGNet, MFCAD++, MFCAD |
| **O-ring** (groove) | AAGNet, MFCAD++ (not present in MFCAD's smaller 16-class set) |
| **Round** (fillet / rounded edge) | AAGNet, MFCAD++ (not present in MFCAD's smaller 16-class set) |
| **Stock / background** (not a machining feature — the unmachined blank material) | AAGNet, MFCAD++, MFCAD |

Seven distinct gap concepts (six real feature-geometry gaps plus the non-feature "stock" background label). None of the three taxonomy-defining references contain any class corresponding to MachinaQ's `boss`, `thread`, or `drill` types either — that asymmetry runs the other direction (MachinaQ types with no source-class equivalent in any of these three works) and is noted here for completeness, though it isn't a "gap" in the sense this section tracks (a gap here means a *source* class with no MachinaQ target).

---

## Canonical Taxonomy Recommendation

**Recommendation: adopt AAGNet's MFInstSeg class list (§ Taxonomy-Defining References → AAGNet) as MachinaQ's long-term canonical target taxonomy**, when a future change evolves `Feature.feature_type` beyond its current five values.

Rationale, traceable to the analysis above:

1. **Superset relationship, verified directly in the primary source.** AAGNet's own `dataset/Utils/parameters.py` defines two label lists: `feat_names` (its full 24-class + stock MFInstSeg taxonomy) and `feat_names_planar`, which is name-for-name and order-for-order identical to the Graph Representation paper's 16-class MFCAD taxonomy (§ Taxonomy-Defining References → Graph Representation paper). AAGNet's own codebase already unifies both taxonomies MachinaQ would otherwise need to reconcile separately — adopting AAGNet's set means MFCAD, MFCAD++, and MFInstSeg are all covered by one target list, not three.
2. **Existing training infrastructure.** MachinaQ's `train_aagnet.py` and `setup_aagnet.bat` already target this exact dataset/taxonomy (see proposal.md's Why) — no other reference here has any MachinaQ-side infrastructure pointed at it.
3. **Richer per-feature output than a flat class list.** Per § Taxonomy-Defining References → AAGNet, AAGNet performs instance segmentation and bottom-face segmentation alongside semantic classification — information MachinaQ's current `Feature` dataclass (a flat `feature_type` + `face_ids` list) doesn't yet capture but could be extended to use, unlike a plain 16- or 24-class label list alone.
4. **The gap list (§ Gap Summary) is the concrete adoption cost.** Seven concepts — chamfer, passage, step, pocket, O-ring, round, and the non-feature stock/background label — would need new `feature_type` values (or a decision to keep them out of scope) if this recommendation is adopted. This is recorded as the specific, itemized scope a future schema-evolution change would need to resolve, not left as a vague "more classes are needed."

Adopting this recommendation — i.e. actually changing `Feature.feature_type`'s values — is explicitly **not** done by this change; see the Scope Boundary at the top of this document.

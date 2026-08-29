## 1. Primary-source research

- [x] 1.1 Pull AAGNet's exact per-face class list and instance/bottom-face segmentation definition from the `whjdark/AAGNet` repo's dataset/label definitions (not from this repo's docstring paraphrase), citing the exact file/section; verify by listing the sourced class names against `train_aagnet.py`'s existing "24 classes" docstring claim and noting any discrepancy found
- [x] 1.2 Pull Hierarchical CADNet's target taxonomy (MFCAD++ class list) from its paper/repo, citing the source; verify the class count and names are recorded with a citation
- [x] 1.3 Pull "Graph Representation of 3D CAD Models for Machining Feature Recognition with Deep Learning"'s classified feature set from its paper, citing the source; verify the class list is recorded with a citation
- [x] 1.4 Confirm BRepNet's evaluated taxonomy/dataset (the MFCAD-family segmentation task it reports results on) from its paper/repo, citing the source; verify the citation names the specific dataset/taxonomy, not just "a segmentation task"
- [x] 1.5 Confirm UV-Net's evaluated tasks and any segmentation taxonomy it reports results on (e.g. face segmentation on MFCAD or similar) from its paper/repo, citing the source; verify the citation names the specific dataset(s)/task(s), or explicitly notes if UV-Net's published results don't include a machining-feature segmentation task

## 2. Mapping tables

- [x] 2.1 Build the AAGNet class table (source class → MachinaQ `feature_type` or `unmapped`) per `specs/gnn-feature-taxonomy-mapping/spec.md`'s "Class-level mapping table" requirement; verify every AAGNet class from task 1.1 appears exactly once in the table
- [x] 2.2 Build the Hierarchical CADNet class table; verify every class from task 1.2 appears exactly once
- [x] 2.3 Build the graph-representation-paper class table; verify every class from task 1.3 appears exactly once
- [x] 2.4 Write the BRepNet and UV-Net sections as representation-technique summaries (not class tables), each stating the taxonomy/dataset they were evaluated against per tasks 1.4/1.5; verify each section explicitly states "representation technique, not an independent taxonomy" per the spec's "Representation-technique reference" scenario

## 3. Gap and recommendation sections

- [x] 3.1 Build the consolidated, deduplicated gap summary from all `unmapped` entries across tasks 2.1-2.3; verify every `unmapped` table entry appears in the summary, deduplicated where the same concept (e.g. "pocket") recurs across references
- [x] 3.2 Write the canonical-taxonomy recommendation section, citing specific findings from the per-reference and gap sections per the spec's "Recommendation is traceable to the analysis" scenario; verify the rationale references at least one specific finding from section 1 and one from section 3.1
- [x] 3.3 Add the explicit scope-boundary statement (no code/runtime behavior changes as a result of this document) per the spec's "No implied runtime change" requirement; verify the statement is present near the top of the document

## 4. Assemble and validate the document

- [x] 4.1 Assemble `docs/GNN_FEATURE_TAXONOMY_MAPPING.md` from sections 1-3 in reading order (per-reference classification → mapping tables → gap summary → recommendation → scope boundary); verify the file exists and every spec requirement's scenario is satisfied by re-checking each against `specs/gnn-feature-taxonomy-mapping/spec.md`
- [x] 4.2 Cross-link the new document from `README.md` (near the existing AAGNet/model-architecture references) so it's discoverable; verify by re-reading the updated `README.md` section

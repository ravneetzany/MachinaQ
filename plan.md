## Plan: AI Model for STEP File Manufacturing Feature Analysis

Build a complete Python-based pipeline using OpenCASCADE (pythonocc) for parsing STEP/.p21 files and extracting B-Rep geometry, combined with PyTorch for deep learning models (PointNet++, 3D CNNs, GNNs) to recognize manufacturing features like holes, bosses, slots, threads, and drills. Output structured JSON and integrate with CAD viewers and CAM systems for manufacturing insights. Leverage NIST STEP File Analyzer and Viewer (SFA) for validation, sample data, and baseline analysis.

**Steps**

### Phase 1: Project Setup and Dependencies
1. Create project structure: src/, data/, models/, tests/, docs/
2. Set up Python environment (venv or conda) on Windows
3. Install dependencies: pythonocc-core (OpenCASCADE), PyTorch, NumPy, Pandas, scikit-learn, matplotlib, trimesh (for point clouds), networkx (for graphs)
4. Configure build tools: requirements.txt, setup.py or pyproject.toml
5. Download and install NIST SFA (v5.41) from GitHub for validation and sample files

### Phase 2: CAD Parsing and Geometry Extraction
1. Implement STEP file loader using pythonocc to read .step/.p21 files
2. Extract B-Rep entities: faces, edges, vertices into data structures (e.g., lists of shapes)
3. Convert geometry to point clouds, meshes, or adjacency graphs for AI input
4. Add error handling for malformed files and logging
5. Validate parsing against NIST SFA outputs (spreadsheets, PMI reports)

### Phase 3: Primitive Shape Classification
1. Detect basic surfaces: planar, cylindrical, conical, helical using geometric properties (normals, curvatures)
2. Build adjacency graph of faces (nodes: faces, edges: shared edges)
3. Implement graph representation using NetworkX
4. Validate classification with unit tests on simple CAD models
5. Cross-check with SFA viewer for PMI and geometric features

### Phase 4: Rule-Based Feature Recognition
1. Define heuristics for features:
   - Hole: cylindrical face penetrating solid (check depth, diameter)
   - Boss: protrusion with cylindrical/conical base
   - Slot: parallel planes with semicircular ends
   - Thread: helical surface on cylinder
   - Drill: hole with conical entry
2. Apply rules to adjacency graph to detect features
3. Output initial feature list with positions, sizes
4. Test on known CAD parts
5. Compare detections with SFA semantic PMI reports

### Phase 5: AI Model Integration
1. Prepare data pipelines for ML: convert geometry to tensors (point clouds for PointNet, voxels for CNNs, graphs for GNNs)
2. Implement PointNet++ for point cloud classification
3. Implement 3D CNN for voxel-based detection
4. Implement GNN for adjacency graph analysis
5. Train models on labeled data (supervised learning for feature classification)
6. Ensemble rule-based and AI detections for robustness
7. Use SFA validation properties for ground truth in training

### Phase 6: Training Data Preparation
1. Download and preprocess Engineering ShapeNet dataset (filter manufacturing parts)
2. Generate synthetic CAD parts using pythonocc: create models with known features (holes, bosses, etc.)
3. Download NIST sample STEP files and CAD models for testing and augmentation
4. Annotate data: label features manually or via automation scripts, using SFA spreadsheets as reference
5. Augment data: rotations, scales, noise for generalization
6. Split into train/val/test sets (aim for 10k samples)

### Phase 7: Output and Integration
1. Generate JSON output with feature counts, types, positions, dimensions
2. Integrate with CAD viewer: highlight features in 3D model (e.g., using pyvista or vtk, inspired by SFA's X3D viewer)
3. Integrate with CAM systems: export feature data for machining paths (e.g., G-code generation)
4. Add API endpoints for web integration and automated validation
5. Document usage and create demo scripts
6. Validate outputs against SFA reports for conformance

**Relevant files**
- `src/parser.py` — STEP loading and geometry extraction
- `src/classifier.py` — Primitive shape detection
- `src/features.py` — Rule-based feature recognition
- `src/models/` — AI model implementations (pointnet.py, cnn.py, gnn.py)
- `data/` — Training datasets, synthetic generators, NIST samples
- `tests/` — Unit and integration tests, including SFA comparisons
- `requirements.txt` — Dependencies
- `nist_sfa/` — Downloaded SFA tool and outputs

**Verification**
1. Unit tests for each module: parse a simple STEP file, classify shapes, detect features
2. Integration test: full pipeline on a test CAD part with known features
3. Accuracy evaluation: precision/recall on test set for feature detection, benchmark against SFA
4. Performance: time to process a STEP file, memory usage
5. Manual verification: visualize outputs in CAD viewer, check CAM integration, compare with SFA viewer

**Decisions**
- Language: Python for ease of AI/ML integration
- Libraries: OpenCASCADE via pythonocc for CAD, PyTorch for deep learning
- Scope: Full implementation from parsing to output
- Data: Use Engineering ShapeNet + generate synthetic data with lots of samples + NIST samples
- Output: JSON + CAD viewer highlighting + CAM system integration
- Platform: Windows-compatible (ensure libraries support Windows)
- Validation: Use NIST SFA as baseline for analysis and conformance
- Resource constraint: GPU not required; optimize models and training for CPU-only environments

**Further Considerations**
1. Data volume: 10k samples target for effective training
2. Hardware: CPU-only training and inference; ensure models and preprocessing are efficient
3. Licensing: Ensure all libraries are compatible with project license (e.g., OpenCASCADE is LGPL)
4. Integration: Provide API endpoints for automated validation against SFA
# MachinaQ

**AI-powered manufacturing feature recognition from STEP/CAD files.**

MachinaQ combines B-Rep graph neural networks, PointNet point cloud classification, and a Claude-powered AI jury to detect and classify 25 machining features — holes, pockets, slots, threads, steps, and more — with structured JSON output ready for CAM and QA integration.

---

## Overview

MachinaQ ingests raw STEP files (`.step`, `.stp`, `.p21`) directly from CAD exports and produces structured feature reports without requiring any external CAD software. The pipeline chains four stages:

1. **Parse** — Text-based STEP parsing extracts geometry primitives and auto-detects units (inches vs. millimeters)
2. **Classify** — B-Rep surface types are identified (planar, cylindrical, conical)
3. **Predict** — Deep learning models classify features and sub-types
4. **Report** — JSON output with feature labels, ASME/ISO standard matches, and confidence scores

An optional **AI Jury** stage uses three Claude agents (Engineer, Inspector, Judge) to review predictions against standards and flag items for human review.

---

## Features

- **Direct STEP parsing** — no OpenCASCADE or CAD kernel required for basic analysis
- **Unit auto-detection** — heuristic + keyword detection of inch vs. metric files
- **25-class GNN segmentation** — full per-face instance segmentation via AAGNet on MFInstSeg
- **PointNet classifiers** — 5-class feature detection + binary through/blind-hole discrimination
- **Unified multi-task model** — joint feature type + hole sub-type prediction
- **ASME / ISO standards library** — built-in lookup tables for clearance holes, tap drills, and thread sizes
- **AI Jury evaluation** — three-agent Claude pipeline for explainability and QA
- **REST API** — FastAPI service for integration with CAM systems and web platforms
- **CPU-optimized** — all inference paths run without a GPU

---

## Architecture

```
STEP File
   │
   ▼
┌──────────────────┐
│  StepTextParser  │  ← Direct text parsing, unit detection
└────────┬─────────┘
         │ primitives (points, circles, cylinders, cones)
         ▼
┌──────────────────┐
│PrimitiveClassifier│  ← B-Rep face typing (planar / cylindrical / conical)
└────────┬─────────┘
         │ classified surfaces
         ▼
┌──────────────────┐
│ FeatureDetector  │  ← Rule-based: holes, bosses, slots, threads, drills
└────────┬─────────┘
         │ candidate features
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌──────────────────┐                    ┌───────────────────┐
│  PointNet /      │                    │  AAGNet (GNN)     │
│  Unified Model   │                    │  25-class segment │
│  (5-class + b/w) │                    │  on B-Rep graph   │
└────────┬─────────┘                    └─────────┬─────────┘
         └───────────────┬──────────────────────┘
                         ▼
               ┌─────────────────┐
               │  ASME/ISO Match │  ← Nearest standard size + tolerance %
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │   AI Jury       │  ← Engineer → Inspector → Judge (Claude)
               │  (optional)     │
               └────────┬────────┘
                        ▼
                  JSON Report
```

### Models

| Model | Classes | Architecture | Input |
|---|---|---|---|
| PointNet | 5 (hole, boss, slot, thread, drill) | 1D conv encoder + FC head | (B, 3, 1024) point cloud |
| PointNetBinary | 2 (through / blind) | PointNet encoder + deeper FC | (B, 3, 1024) point cloud |
| MachinaQUnified | 5 + 2 (multi-task) | Shared encoder, dual heads | (B, 3, 1024) point cloud |
| AAGNet | 25 (MFInstSeg taxonomy) | Graph neural network | B-Rep adjacency graph |

---

## Installation

### Requirements

- Python 3.10+
- Windows (primary target); Linux supported for training
- 8 GB RAM minimum; 32 GB recommended for GNN dataset loading
- GPU optional — all models run on CPU

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/MachinaQ.git
cd MachinaQ

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate      # Windows PowerShell
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

#### Conda (recommended for GNN training on Windows)

```batch
call E:\ProgramFilez\Anaconda\Scripts\activate.bat pyoccMachinaQ
set DGLBACKEND=pytorch
cd E:\AiTools\AIModel_StepAnalyze
```

Key conda environment specs:
- Python 3.10.9
- PyTorch 2.3.1+cu121
- DGL 2.2.1 (backend: PyTorch)

#### Optional: OpenCASCADE bindings

`pythonocc-core` enables full B-Rep face classification. Without it the pipeline falls back to rule-based detection.

```bash
conda install -c conda-forge pythonocc-core
```

---

## Running

### API Server

```bash
uvicorn src.api:app --reload
```

- API root: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze` | Parse STEP file → features + model predictions |
| `POST` | `/evaluate` | Run AI Jury on a prediction report |
| `POST` | `/validate` | Compare predictions against an expected report |

**Example — analyze a STEP file:**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_path": "nist_sfa/nist_ctc_01_asme1_rd.stp"}'
```

**Example response (abbreviated):**

```json
{
  "file": "nist_ctc_01_asme1_rd.stp",
  "unit_system": "inches",
  "features": [
    {
      "id": 1,
      "type": "hole",
      "subtype": "through",
      "radius_in": 0.1875,
      "asme_label": "#12 Clearance",
      "asme_error_pct": 0.2,
      "confidence": 0.97
    }
  ]
}
```

---

## Training

All training is launched from the project root via `run_train.py`:

```bash
python run_train.py --model <model_type> [options]
```

### Model types

| `--model` | What it trains | Dataset |
|---|---|---|
| `pointnet` | 5-class feature classifier | NIST .stp files + `nist_sfa/holeTrain/` |
| `through-hole` | Binary through/blind classifier | Synthetic cylinder + disk samples |
| `unified` | Multi-task (feature + through/blind) | Combined above; use `--merge-weights` to init from pre-trained |
| `gnn` | 25-class AAGNet segmentation | MFInstSeg (~20 GB, 14 shards) |

### Common options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | model default | Override epoch count |
| `--batch-size` | model default | Override batch size |
| `--lr` | `1e-3` | Override learning rate |
| `--aug` | `15` | Augmentation factor per sample (PointNet variants) |
| `--merge-weights` | off | Init unified model from pre-trained PointNet + through-hole weights |

### Examples

```bash
# Train PointNet for 20 epochs
python run_train.py --model pointnet --epochs 20 --batch-size 16

# Train binary through/blind classifier with heavy augmentation
python run_train.py --model through-hole --aug 20

# Fine-tune unified model starting from pre-trained component weights
python run_train.py --model unified --merge-weights

# Train GNN on MFInstSeg (plan for 20–60 min dataset load time)
python run_train.py --model gnn
```

Training artifacts are saved to `outputs/`:
- Logs: `outputs/<model>_train.log`
- Best checkpoint: `outputs/machinaq_<model>.pth`

---

## AI Jury

The AI Jury is an optional evaluation layer that routes predictions through three Claude agents:

| Agent | Role | Output |
|---|---|---|
| **Engineer** | Advocates for the model's prediction from a manufacturing perspective | Plausibility, tooling fit, confidence boost |
| **Inspector** | Audits against ASME/ISO standards | Standards compliance, violations, severity |
| **Judge** | Arbitrates the debate and issues a final verdict | Verdict, score, human-review flag |

**Verdict levels:** `APPROVED` · `APPROVED_WITH_CAUTION` · `NEEDS_REVIEW` · `REJECTED`

Set your Anthropic API key before running the jury:

```bash
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python src/evaluate_with_agents.py
```

---

## Project Structure

```
MachinaQ/
├── src/
│   ├── api.py                  # FastAPI application
│   ├── pipeline.py             # Main orchestration
│   ├── parser.py               # STEP text parser & unit detection
│   ├── primitive.py            # B-Rep surface classification
│   ├── features.py             # Rule-based feature detection
│   ├── asme_standards.py       # ASME / ISO lookup tables
│   ├── hole_visualizer.py      # 3D visualization utilities
│   ├── evaluate_with_agents.py # AI Jury orchestration
│   ├── agents/
│   │   ├── engineer_agent.py
│   │   ├── inspector_agent.py
│   │   └── judge_agent.py
│   └── train*.py               # Individual training scripts
├── models/
│   ├── pointnet.py             # PointNet architectures
│   ├── machinaq_unified.py     # Unified multi-task model
│   └── pointnet_trained.pth    # Pre-trained PointNet weights (3.2 MB)
├── machgnn/                    # AAGNet GNN subproject
│   └── dataset/MFInstSeg/      # 25-class segmentation dataset
├── nist_sfa/                   # NIST sample STEP files & validation outputs
│   └── holeTrain/              # Synthetic hole dataset for PointNet
├── data/                       # Additional training data
├── outputs/                    # Training logs and checkpoints
├── run_train.py                # Training launcher
├── requirements.txt
└── TRAINING_HELP.md            # Environment setup & troubleshooting
```

---

## Standards Reference

MachinaQ includes built-in lookup tables for:

| Standard | Scope |
|---|---|
| ASME B18.2.8 | Clearance hole diameters (#0 – 1-1/2 in) |
| ASME B1.1 | Tap drill sizes (UNC/UNF threads) |
| ASME B94.11M | Standard drill bit sizes |
| ISO 273 | Metric clearance holes — fine / medium / coarse |
| ISO 965-1 | Metric tap drills (M1 – M64) |

Matching uses a ±5% snap tolerance by default. Each matched feature reports the standard label and deviation percentage.

---

## Troubleshooting

**DGL import error on Windows**
```bash
set DGLBACKEND=pytorch
python -c "import dgl; print(dgl.__version__)"
```

**pythonocc not found**
The pipeline degrades gracefully to rule-based detection. To enable full B-Rep support, install via conda (pip wheels are unstable on Windows).

**GNN dataset loading takes 20–60 minutes**
This is normal for the 14-shard MFInstSeg dataset on first load. Subsequent runs use cached partitions.

**CUDA not available**
All models run on CPU. Training is slower but fully supported. Expected: ~2 min/epoch on CPU vs ~30 sec/epoch on GPU.

---

## License

MIT

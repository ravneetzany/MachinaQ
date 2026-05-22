# MachinaQ

AI-powered manufacturing feature recognition from STEP/CAD files. MachinaQ combines B-Rep graph neural networks and point cloud classification to detect and classify 25 machining features — holes, pockets, slots, threads, steps, and more — with structured JSON output for CAM and QA integration.

## Getting Started

1. Create a Python virtual environment on Windows.
2. Install dependencies from `requirements.txt`.
3. Run the API with `uvicorn src.api:app --reload`.

## Project Layout

- `src/` — pipeline modules and API implementation
- `data/` — STEP datasets and generated samples
- `models/` — saved ML models and checkpoints
- `tests/` — unit tests
- `nist_sfa/` — NIST STEP File Analyzer resources and validation outputs

## Notes

- The system uses direct STEP text parsing for geometry extraction.
- Optimized for CPU-only training and inference.

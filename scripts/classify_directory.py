#!/usr/bin/env python3
"""Batch-classify a directory of `.scad` / FreeCAD `.py` part scripts into
required CNC operations, writing one JSON report per discovered part.

Usage:
    python scripts/classify_directory.py <library-root> [--out <output-dir>]

Example:
    python scripts/classify_directory.py /home/ravneetzany/projects/openscad-parts-library
    python scripts/classify_directory.py /home/ravneetzany/projects/freecad-parts-library --out outputs/freecad_reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features import Feature  # noqa: E402
from src.operation_classifier import classify_features, summarize_part  # noqa: E402
from src.primitive import SurfacePrimitive  # noqa: E402
from src.py_source_ingest import ingest_directory as ingest_py_directory  # noqa: E402
from src.scad_ingest import ingest_directory as ingest_scad_directory  # noqa: E402


def _passthrough_features(primitives: List[SurfacePrimitive]) -> List[Feature]:
    """One primitive == one feature at this ingestion path's granularity —
    `.scad`/`.py` ingestion already emits one coarse primitive per shape
    call, unlike STEP B-Rep faces, which need `FeatureDetector`'s
    adjacency-evidence rules to group faces into named features. See
    design.md decision 4's "Correction found during implementation"."""
    return [Feature(feature_type=p.type, face_ids=[p.face_id], parameters={}) for p in primitives]


def _report_for_result(part_name: str, result) -> Dict[str, Any]:
    primitives: List[SurfacePrimitive] = result.primitives
    features = _passthrough_features(primitives)
    unclassified_face_ids: List[int] = []

    feature_operations = classify_features(features, primitives, result.principal_axis)
    operations_summary = summarize_part(feature_operations)

    return {
        "part_name": part_name,
        "primitives": [
            {"face_id": p.face_id, "type": p.type, "details": p.details} for p in primitives
        ],
        "principal_axis": (
            None if result.principal_axis is None
            else {
                "direction": result.principal_axis.direction,
                "point": result.principal_axis.point,
            }
        ),
        "features": [
            {
                "feature_type": f.feature_type,
                "face_ids": f.face_ids,
                "operation": op.operation,
                "operation_rationale": op.rationale,
            }
            for f, op in zip(features, feature_operations)
        ],
        "unclassified_face_ids": unclassified_face_ids,
        "operations_summary": {
            "primary_process": operations_summary.primary_process,
            "secondary_processes": operations_summary.secondary_processes,
            "rationale": operations_summary.rationale,
        },
    }


def classify_directory(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for entry in ingest_scad_directory(root):
        part_name = entry.path.stem
        if entry.error is not None:
            print(f"[unparsed] {entry.path}: {entry.error}")
            skipped += 1
            continue
        report = _report_for_result(part_name, entry.result)
        report["source_file"] = str(entry.path)
        (out_dir / f"{part_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        written += 1

    for entry in ingest_py_directory(root):
        part_name = entry.path.stem
        if entry.error is not None:
            print(f"[unparsed] {entry.path}: {entry.error}")
            skipped += 1
            continue
        report = _report_for_result(part_name, entry.result)
        report["source_file"] = str(entry.path)
        (out_dir / f"{part_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        written += 1

    print(f"\n{written} report(s) written to {out_dir}, {skipped} file(s) unparsed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library_root", type=Path, help="Directory to scan for .scad/.py part files")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for JSON reports (default: outputs/classify_reports/<library-name>)",
    )
    args = parser.parse_args()

    out_dir = args.out or (ROOT / "outputs" / "classify_reports" / args.library_root.name)
    classify_directory(args.library_root, out_dir)


if __name__ == "__main__":
    main()

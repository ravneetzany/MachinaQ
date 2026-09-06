"""ASCII STL ingestion for point-cloud training.

Lightweight text parser — no trimesh/OpenCASCADE dependency, consistent with
this project's StepTextParser approach of reading CAD exports directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def parse_ascii_stl_vertices(path: Union[str, Path]) -> np.ndarray:
    """Extract all triangle vertices from an ASCII STL file.

    Returns an (M, 3) float32 array, M = 3 * facet_count. Raises ValueError
    if the file isn't ASCII STL (binary STL isn't needed for this dataset —
    see nist_sfa/stl, which ships ASCII-only).
    """
    verts = []
    with open(path, "r", errors="strict") as f:
        first = f.readline()
        if not first.lstrip().startswith("solid"):
            raise ValueError(f"{path}: not an ASCII STL file")
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError(f"{path}: no vertices found")
    return np.array(verts, dtype=np.float32)


def sample_point_cloud(
    vertices: np.ndarray, n_points: int = 1024, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Resample a vertex array to a fixed-size point cloud.

    Subsamples without replacement when there are enough vertices, otherwise
    samples with replacement to pad up to n_points (small facet counts are
    common in this dataset — see stl_ingest tests).
    """
    rng = rng or np.random.default_rng()
    n = len(vertices)
    if n >= n_points:
        idx = rng.choice(n, size=n_points, replace=False)
    else:
        idx = rng.choice(n, size=n_points, replace=True)
    return vertices[idx]


def center_and_scale(points: np.ndarray) -> np.ndarray:
    """Center at centroid and scale to fit the unit sphere."""
    pts = points - points.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-8
    return (pts / scale).astype(np.float32)

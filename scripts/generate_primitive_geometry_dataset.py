"""Generate parametric variations of the 10 primitive-geometry classes
(cone/cylinder/polygon/sphere/wedge x boss/pocket) as ASCII STL files, to
supplement the single hand-made FreeCAD example per class in
data/primitive_geometry_stl/<N>_<name>/.

The original 10 hand-made files (one per class) are real ground truth from
the user's FreeCAD parts and are left untouched; this script only adds more
`gen_*.stl` files alongside them, randomizing size, XY position on the base
pad, and (for polygon/wedge) yaw rotation, so run_train.py's
`primitive-geometry` target has enough per-class variety to learn general
shape features instead of memorizing one exact point cloud.

Requires pythonocc-core (OCC.Core) — this repo's .venv has it built from
source (see the FreeCAD-tooling training session that added this script).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Trsf, gp_Vec
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeWedge,
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import TopoDS_Shape

LINEAR_DEFLECTION = 0.3  # matches the original hand-made STL exports

ROOT = Path(__file__).parent.parent
OUT_ROOT = ROOT / "data" / "primitive_geometry_stl"

CLASS_DIRS = {
    "cone_boss": "0_cone_boss",
    "cone_pocket": "1_cone_pocket",
    "cylinder_boss": "2_cylinder_boss",
    "cylinder_pocket": "3_cylinder_pocket",
    "polygon_boss": "4_polygon_boss",
    "polygon_pocket": "5_polygon_pocket",
    "sphere_boss": "6_sphere_boss",
    "sphere_pocket": "7_sphere_pocket",
    "wedge_boss": "8_wedge_boss",
    "wedge_pocket": "9_wedge_pocket",
}

BossOrPocket = str  # "boss" | "pocket"


def _bbox(shape: TopoDS_Shape) -> Tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()


def _rotate(shape: TopoDS_Shape, axis_dir: Tuple[float, float, float], angle_rad: float) -> TopoDS_Shape:
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*axis_dir)), angle_rad)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _translate(shape: TopoDS_Shape, dx: float, dy: float, dz: float) -> TopoDS_Shape:
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def make_pad(dx: float, dy: float, dz: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), dx, dy, dz).Shape()


def build_tool(shape_kind: str, rng: random.Random) -> Tuple[TopoDS_Shape, float]:
    """Build one tool shape natively spanning z in [0, H] (wide/base face at
    z=0, extending +z) — a uniform "grows upward from its base" convention
    so the caller can flip it for pockets and center/translate it uniformly
    regardless of shape type. Returns (shape, max_extent) — max_extent is
    used by the caller to keep the feature within the pad's margins.
    """
    if shape_kind == "cylinder":
        r = rng.uniform(8, 18)
        h = rng.uniform(10, 22)
        return BRepPrimAPI_MakeCylinder(r, h).Shape(), r
    if shape_kind == "cone":
        r1 = rng.uniform(8, 18)
        r2 = rng.uniform(0, r1 * 0.6)
        h = rng.uniform(10, 22)
        return BRepPrimAPI_MakeCone(r1, r2, h).Shape(), r1
    if shape_kind == "polygon":
        dxf = rng.uniform(16, 32)
        dyf = rng.uniform(16, 32)
        dzf = rng.uniform(10, 22)
        shape = BRepPrimAPI_MakeBox(dxf, dyf, dzf).Shape()
        yaw = rng.uniform(0, 2 * math.pi)
        shape = _rotate(shape, (0, 0, 1), yaw)
        return shape, max(dxf, dyf) * 0.75
    if shape_kind == "wedge":
        # Checked the real hand-made wedgeBoss.FCStd directly: its
        # PartDesign::AdditiveWedge Placement has Yaw-Pitch-Roll=(0,0,0) —
        # no rotation at all, just a Z translation onto the pad's top face.
        # So native Z (0..dzf) IS already the vertical "up" axis (matching
        # cone/cylinder/box's "grows upward from base at z=0" convention
        # directly, no extra rotation needed), and native Y (0..dyf) is the
        # taper direction, which runs HORIZONTALLY across the pad's top
        # face, not vertically. An earlier version of this generator rotated
        # the wedge +90 about X believing the taper direction should become
        # vertical (like a pyramid narrowing as it rises) — that produced a
        # systematically different, wrong shape and was the actual cause of
        # a persistent, high-confidence wedge_boss -> polygon_boss
        # misclassification across multiple retrainings (a real generator
        # bug, not training variance or a footprint-scale issue).
        dxf = rng.uniform(8, 24)   # width (X), horizontal
        dyf = rng.uniform(10, 24)  # taper length (Y), horizontal
        dzf = rng.uniform(8, 24)   # height (Z), vertical — native, unrotated
        # Centered on the real example's ~0.6 taper (not at a range edge —
        # a range like [0.2, 0.6] made 0.6 the least-typical value in its
        # own class, and the model misclassified the real file even via its
        # own native STL, not just through the STEP/OCC round-trip).
        taper_x = rng.uniform(0.35, 0.75)
        taper_z = rng.uniform(0.35, 0.75)
        top_w = dxf * taper_x
        top_h = dzf * taper_z
        xmin = (dxf - top_w) / 2.0
        xmax = xmin + top_w
        zmin = (dzf - top_h) / 2.0
        zmax = zmin + top_h
        shape = BRepPrimAPI_MakeWedge(dxf, dyf, dzf, xmin, zmin, xmax, zmax).Shape()
        yaw = rng.uniform(0, 2 * math.pi)
        shape = _rotate(shape, (0, 0, 1), yaw)
        return shape, max(dxf, dyf) * 0.75
    raise ValueError(f"unknown shape_kind: {shape_kind}")


def place_tool(
    shape: TopoDS_Shape,
    op: BossOrPocket,
    cx: float,
    cy: float,
    pad_top_z: float,
) -> TopoDS_Shape:
    """Center the tool's XY footprint at (cx, cy) and anchor its base face
    to pad_top_z — flipped to grow downward for a pocket cut."""
    if op == "pocket":
        shape = _rotate(shape, (1, 0, 0), math.pi)  # base stays at local z=0, tip now grows -z
    xmin, ymin, _zmin, xmax, ymax, _zmax = _bbox(shape)
    cx0 = (xmin + xmax) / 2
    cy0 = (ymin + ymax) / 2
    return _translate(shape, cx - cx0, cy - cy0, pad_top_z)


def place_sphere(shape: TopoDS_Shape, op: BossOrPocket, cx: float, cy: float, pad_top_z: float, r: float, rng: random.Random) -> TopoDS_Shape:
    """Sphere is symmetric about its own center, so boss vs pocket only
    differs in the boolean op (Fuse vs Cut) — same placement logic covers
    both, just with a random vertical offset controlling how much of the
    sphere protrudes/embeds."""
    offset_frac = rng.uniform(-0.3, 0.3)
    return _translate(shape, cx, cy, pad_top_z + offset_frac * r)


def generate_sample(class_name: str, rng: random.Random) -> TopoDS_Shape:
    shape_kind, op = class_name.rsplit("_", 1)
    pad_dx = rng.uniform(110, 150)
    pad_dy = rng.uniform(80, 110)
    pad_dz = 20.0 if op == "boss" else 60.0
    pad = make_pad(pad_dx, pad_dy, pad_dz)

    if shape_kind == "sphere":
        r = rng.uniform(10, 20)
        margin = r + 10
        cx = rng.uniform(margin, pad_dx - margin)
        cy = rng.uniform(margin, pad_dy - margin)
        tool_raw = BRepPrimAPI_MakeSphere(r).Shape()
        tool = place_sphere(tool_raw, op, cx, cy, pad_dz, r, rng)
    else:
        tool_raw, half_extent = build_tool(shape_kind, rng)
        margin = half_extent + 10
        cx = rng.uniform(margin, pad_dx - margin)
        cy = rng.uniform(margin, pad_dy - margin)
        tool = place_tool(tool_raw, op, cx, cy, pad_dz)

    if op == "boss":
        return BRepAlgoAPI_Fuse(pad, tool).Shape()
    return BRepAlgoAPI_Cut(pad, tool).Shape()


def write_ascii_stl(shape: TopoDS_Shape, out_path: Path) -> None:
    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION)
    writer = StlAPI_Writer()
    writer.SetASCIIMode(True)
    writer.Write(shape, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for class_name, dir_name in CLASS_DIRS.items():
        class_dir = OUT_ROOT / dir_name
        class_dir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        attempts = 0
        while n_ok < args.samples_per_class and attempts < args.samples_per_class * 3:
            attempts += 1
            try:
                shape = generate_sample(class_name, rng)
                xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
                if (xmax - xmin) < 1 or (ymax - ymin) < 1 or (zmax - zmin) < 1:
                    continue  # degenerate boolean result, retry
                out_path = class_dir / f"gen_{n_ok:03d}.stl"
                write_ascii_stl(shape, out_path)
                n_ok += 1
            except Exception as e:
                print(f"  [{class_name}] sample failed, retrying: {e}")
        print(f"{dir_name}: {n_ok}/{args.samples_per_class} generated")


if __name__ == "__main__":
    main()

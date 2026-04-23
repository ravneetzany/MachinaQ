"""Primitive surface classification for STEP geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import os
    import sys
    # Add OCCT DLLs to PATH for Windows
    if sys.platform == 'win32':
        occ_bin_path = os.path.join(os.path.dirname(sys.executable), '..', 'Library', 'bin')
        occ_bin_path = os.path.abspath(occ_bin_path)
        if occ_bin_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] = occ_bin_path + os.pathsep + os.environ.get('PATH', '')
    
    import OCC
    from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_Surface
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import TopoDS_Face
    from OCC.Core.BRepTools import breptools_UVBounds
    OCC_AVAILABLE = True
except Exception:
    Geom_Plane = Geom_CylindricalSurface = Geom_ConicalSurface = Geom_Surface = None  # type: ignore
    BRep_Tool = None  # type: ignore
    TopoDS_Face = None  # type: ignore
    breptools_UVBounds = None  # type: ignore
    OCC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SurfacePrimitive:
    face_id: int
    type: str
    details: Dict[str, float]


class PrimitiveClassifier:
    def __init__(self) -> None:
        if not OCC_AVAILABLE:
            logger.warning("pythonocc-core is not available. Primitive classification will use placeholders.")

    def classify_face(self, face: "TopoDS_Face", face_id: int) -> SurfacePrimitive:
        if OCC_AVAILABLE and BRep_Tool is not None:
            surf = BRep_Tool.Surface(face)
            return self._classify_surface(surf, face_id)

        return SurfacePrimitive(face_id=face_id, type="unknown", details={})

    def _classify_surface(self, surf: "Geom_Surface", face_id: int) -> SurfacePrimitive:
        if isinstance(surf, Geom_Plane):
            return SurfacePrimitive(face_id=face_id, type="planar", details={})
        if isinstance(surf, Geom_CylindricalSurface):
            radius = surf.Radius()
            return SurfacePrimitive(face_id=face_id, type="cylindrical", details={"radius": radius})
        if isinstance(surf, Geom_ConicalSurface):
            return SurfacePrimitive(face_id=face_id, type="conical", details={"semi_angle": surf.SemiAngle()})

        return SurfacePrimitive(face_id=face_id, type="unknown", details={})

    def classify_faces(self, faces: list["TopoDS_Face"]) -> list[SurfacePrimitive]:
        results = []
        for idx, face in enumerate(faces, start=1):
            results.append(self.classify_face(face, idx))
        return results

    def features_for_face(self, primitive: SurfacePrimitive) -> Dict[str, bool]:
        return {
            "is_planar": primitive.type == "planar",
            "is_cylindrical": primitive.type == "cylindrical",
            "is_conical": primitive.type == "conical",
            "is_helical": primitive.type == "helical",
        }

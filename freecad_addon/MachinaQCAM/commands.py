"""The MachinaQ_ClassifyFeature FreeCAD command.

Exports the active document/selection, submits it to MachinaQ's /analyze
API, and shows the resulting feature/operation classification (plus a
best-effort face correlation, when specific faces are selected) in a
read-only task panel.

FreeCAD-only module: relies on FreeCAD/FreeCADGui/Part/PySide, which are
only importable from inside FreeCAD's own Python. `http_client.py`,
`config.py`, and `face_correlation.py` (this command's dependencies) are
themselves FreeCAD-independent and separately unit-tested.
"""

from __future__ import annotations

import os
import tempfile

import FreeCAD
import FreeCADGui as Gui
import Part

import config
import http_client
from face_correlation import correlate_faces
from task_panel import ClassificationTaskPanel

_ICON_PATH = os.path.join(os.path.dirname(__file__), "Resources", "icons", "machinaq_classify.svg")


def _export_target():
    """Return (shape_to_export, selected_faces) — the selected Body if one is
    selected, else the whole active document; plus any individually-selected
    Faces (for the face-correlation section), independent of which was used
    for export."""
    selection = Gui.Selection.getSelectionEx()
    selected_faces = []
    selected_bodies = []

    for sel in selection:
        for sub_name, sub_obj in zip(sel.SubElementNames, sel.SubObjects):
            if sub_name.startswith("Face"):
                selected_faces.append(sub_obj)
        if sel.Object is not None and not sel.SubElementNames:
            selected_bodies.append(sel.Object)

    if selected_bodies:
        shape = selected_bodies[0].Shape
    elif FreeCAD.ActiveDocument is not None:
        shapes = [obj.Shape for obj in FreeCAD.ActiveDocument.Objects if hasattr(obj, "Shape")]
        shape = Part.makeCompound(shapes) if shapes else None
    else:
        shape = None

    return shape, selected_faces


class MachinaQClassifyCommand:
    """Registered as `MachinaQ_ClassifyFeature` via Gui.addCommand."""

    def GetResources(self):
        return {
            "Pixmap": _ICON_PATH,
            "MenuText": "Classify Feature (MachinaQ)",
            "ToolTip": "Classify machining features and required CNC operations via MachinaQ",
        }

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        shape, selected_faces = _export_target()
        if shape is None:
            FreeCAD.Console.PrintError("MachinaQ: no active document to classify.\n")
            return

        temp_dir = tempfile.mkdtemp(prefix="machinaq_")
        step_path = os.path.join(temp_dir, "export.step")
        try:
            shape.exportStep(step_path)
        except Exception as exc:
            FreeCAD.Console.PrintError(f"MachinaQ: failed to export STEP file: {exc}\n")
            return

        api_url = config.get_api_url()
        try:
            report = http_client.classify(step_path, api_url=api_url)
        except http_client.MachinaQUnreachableError as exc:
            FreeCAD.Console.PrintError(
                f"MachinaQ: could not reach the API at {api_url}: {exc}\n"
            )
            return
        except http_client.MachinaQErrorResponse as exc:
            FreeCAD.Console.PrintError(f"MachinaQ: {exc}\n")
            return

        correlations = []
        if selected_faces:
            correlations = correlate_faces(selected_faces, report.get("primitives", []))

        panel = ClassificationTaskPanel(report, correlations)
        Gui.Control.showDialog(panel)


def register() -> None:
    if "MachinaQ_ClassifyFeature" not in Gui.listCommands():
        Gui.addCommand("MachinaQ_ClassifyFeature", MachinaQClassifyCommand())

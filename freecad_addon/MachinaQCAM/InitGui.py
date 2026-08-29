"""MachinaQCAM addon entry point.

Registers the `MachinaQ_ClassifyFeature` command and adds it to the CAM
workbench's toolbar — not a new standalone workbench.

Per design.md decision 2 (revised): uses FreeCAD main window's
`workbenchActivated` Qt signal, polled into existence via a QTimer exactly
the way FreeCAD's own bundled `Tux/PersistentToolbarsGui.py` addon does
(verified against that real, installed, working addon in this environment
— see `/usr/lib/freecad/Mod/Tux/PersistentToolbarsGui.py`).

**Correction from an earlier version:** this used to monkey-patch
`CAMWorkbench.Initialize` (`Initialize()` runs exactly once per session, on
first activation), on the assumption our `InitGui.py` would always run
before CAM's workbench was first activated. That's wrong whenever CAM is
already the active/default workbench at FreeCAD startup — `Initialize()`
already ran before the patch could be installed, so the toolbar was never
added, with no error (the patch installation itself succeeded; it just
patched a method that would never be called again). The
`workbenchActivated` signal fires on every activation, including the
already-active workbench once we explicitly check it right after
connecting, so it doesn't depend on load-order timing at all.
"""

import os
import sys

_ADDON_DIR = os.path.dirname(__file__)
if _ADDON_DIR not in sys.path:
    sys.path.insert(0, _ADDON_DIR)

import FreeCAD
import FreeCADGui as Gui
from PySide import QtCore

import commands

_TOOLBAR_NAME = "MachinaQ"
_COMMAND_NAME = "MachinaQ_ClassifyFeature"
_CAM_WORKBENCH_CLASS_NAME = "CAMWorkbench"

_toolbar_added = False
_timer = QtCore.QTimer()
_mw = Gui.getMainWindow()


def _add_toolbar_if_cam_active() -> None:
    global _toolbar_added
    if _toolbar_added:
        return
    try:
        active = Gui.activeWorkbench()
    except Exception:
        return
    if active is None or active.__class__.__name__ != _CAM_WORKBENCH_CLASS_NAME:
        return
    try:
        active.appendToolbar(_TOOLBAR_NAME, [_COMMAND_NAME])
        _toolbar_added = True
    except Exception as exc:
        FreeCAD.Console.PrintWarning(f"MachinaQCAM: could not append toolbar: {exc}\n")


def _on_workbench_activated(_name=None) -> None:
    _add_toolbar_if_cam_active()


def _start_when_ready() -> None:
    """Poll (like Tux/PersistentToolbarsGui.py does) until the main window's
    event loop and `workbenchActivated` signal actually exist, then connect
    and check the currently-active workbench immediately."""
    if not _mw.property("eventLoop"):
        return
    try:
        _mw.workbenchActivated
    except AttributeError:
        return

    _timer.stop()
    _on_workbench_activated()  # catch the case where CAM is already active
    _mw.workbenchActivated.connect(_on_workbench_activated)


commands.register()
_timer.timeout.connect(_start_when_ready)
_timer.start(500)

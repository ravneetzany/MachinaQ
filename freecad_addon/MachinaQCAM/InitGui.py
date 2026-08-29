"""MachinaQCAM addon entry point.

Registers MachinaQ as its own **standalone FreeCAD workbench** — not
injected into CAM's toolbar.

**Why this differs from the original design:** the original design (see
design.md decisions 2's history) tried to add the classify command
directly into CAM's own toolbar, across four separate fix attempts and
three independent mechanisms (patching `CAMWorkbench.Initialize`, a
`workbenchActivated` signal + polling `QTimer`, and finally a
synchronous-attempt-plus-layered-fallbacks version). Each fix addressed a
real, confirmed bug, and each still resulted in the toolbar intermittently
not appearing in real testing, for reasons that could not be conclusively
identified without live interactive debugging access (not available in
the environment that built this addon). The common thread every time was
that individual pieces (`Gui.activeWorkbench()`, `appendToolbar(...)`,
`Gui.getMainWindow()`) always worked when called directly — only the
*automatic, deferred* wiring into an existing workbench was unreliable.

A standalone workbench sidesteps this entirely: `Gui.addWorkbench(...)` is
the same mechanism CAM (and every other built-in FreeCAD module) uses for
*itself* — there is no "inject into someone else's toolbar after they've
already initialized" timing problem, because MachinaQ owns its own
`Initialize()`, called by FreeCAD exactly the way it calls every other
workbench's. Trade-off: the "Classify Feature" button now lives in its own
"MachinaQ" workbench (shown in the workbench dropdown next to CAM,
PartDesign, etc.), not literally inside CAM's own toolbar as originally
requested.
"""

import os
import sys

_ADDON_DIR = os.path.dirname(__file__)
if _ADDON_DIR not in sys.path:
    sys.path.insert(0, _ADDON_DIR)

import FreeCADGui as Gui

_ICON_PATH = os.path.join(_ADDON_DIR, "Resources", "icons", "machinaq_classify.svg")
_COMMAND_NAME = "MachinaQ_ClassifyFeature"


class MachinaQWorkbench(Workbench):  # noqa: F821 — `Workbench` is a FreeCAD-injected global in InitGui.py, matching every built-in workbench (verified: CAM, PartDesign)
    MenuText = "MachinaQ"
    ToolTip = "MachinaQ feature/operation classification"
    Icon = _ICON_PATH

    def Initialize(self):
        import commands
        commands.register()
        self.appendToolbar("MachinaQ", [_COMMAND_NAME])
        self.appendMenu("MachinaQ", [_COMMAND_NAME])

    def Activated(self):
        pass

    def Deactivated(self):
        pass

    def GetClassName(self):
        return "Gui::PythonWorkbench"


Gui.addWorkbench(MachinaQWorkbench())

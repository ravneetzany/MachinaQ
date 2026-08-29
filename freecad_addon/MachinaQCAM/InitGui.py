"""MachinaQCAM addon entry point.

Registers MachinaQ as its own **standalone FreeCAD workbench** — not
injected into CAM's toolbar.

**Why this differs from the original design:** the original design tried to
add the classify command directly into CAM's own toolbar, across four
separate fix attempts and three independent mechanisms. Each fix addressed
a real, confirmed bug, and each still resulted in the toolbar intermittently
not appearing in real testing. A standalone workbench sidesteps this
entirely: `Gui.addWorkbench(...)` is the same mechanism CAM (and every
other built-in FreeCAD module) uses for *itself* — no "inject into someone
else's already-initialized toolbar" timing problem. Trade-off: the
"Classify Feature" button lives in its own "MachinaQ" workbench (workbench
dropdown), not literally inside CAM's own toolbar as originally requested.
See design.md decision 2 for the full multi-round debugging history.

**Correction found in this round:** the first standalone-workbench version
used a bare, unqualified `Workbench` as the base class, assuming FreeCAD
injects it as a global into every `InitGui.py`'s namespace — true for
FreeCAD's own bundled Mod scripts (verified: CAM's and PartDesign's own
`InitGui.py` both use it bare), but apparently *not* guaranteed for addons
loaded from the user Mod directory. That produced a `NameError` at the
class statement, halting the script before `Gui.addWorkbench(...)` at the
bottom ever ran — matching exactly what live diagnosis found: `commands`
importable (proving only the file's top, before the class, executed) but
the workbench absent from `Gui.listWorkbenches()`, no visible error.
Fixed by resolving the base class defensively (`Gui.Workbench` first, a
bare global as fallback) and wrapping registration in a try/except that
prints to `FreeCAD.Console` on failure — so if this still doesn't work,
there is at least a visible diagnostic instead of total silence.
"""

import os
import sys

import FreeCAD
import FreeCADGui as Gui

# **Correction found via live testing (five rounds in):** FreeCAD executes
# InitGui.py via exec() without setting `__file__` in the namespace — using
# it (as every earlier version of this file did) raises NameError at this
# exact line, silently halting the whole script before anything else runs.
# This explains why InitGui.py never showed up in sys.modules either (same
# symptom, consistent all along). FreeCAD does know the real path — it's
# used as the synthetic filename in tracebacks — it just isn't exposed as a
# namespace global. Resolved instead via FreeCAD's own reported Mod
# locations (user AppData dir, then the system AppHomePath as a fallback),
# checking which one actually contains this addon's files.
def _resolve_addon_dir() -> str:
    candidates = []
    try:
        candidates.append(os.path.join(FreeCAD.getUserAppDataDir(), "Mod", "MachinaQCAM"))
    except Exception:
        pass
    try:
        candidates.append(os.path.join(FreeCAD.ConfigGet("AppHomePath"), "Mod", "MachinaQCAM"))
    except Exception:
        pass
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "commands.py")):
            return candidate
    # Last resort: whichever sys.path entry looks like this addon's folder.
    for entry in sys.path:
        if entry.rstrip("/").endswith("MachinaQCAM") and os.path.isfile(os.path.join(entry, "commands.py")):
            return entry
    raise RuntimeError("MachinaQCAM: could not determine the addon's own directory")


_ADDON_DIR = _resolve_addon_dir()
if _ADDON_DIR not in sys.path:
    sys.path.insert(0, _ADDON_DIR)

_ICON_PATH = os.path.join(_ADDON_DIR, "Resources", "icons", "machinaq_classify.svg")
_COMMAND_NAME = "MachinaQ_ClassifyFeature"

_WorkbenchBase = getattr(Gui, "Workbench", None) or globals().get("Workbench")

if _WorkbenchBase is None:
    FreeCAD.Console.PrintError(
        "MachinaQCAM: could not resolve a Workbench base class "
        "(neither Gui.Workbench nor a bare 'Workbench' global is available) "
        "— the MachinaQ workbench was not registered.\n"
    )
else:
    class MachinaQWorkbench(_WorkbenchBase):
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

    try:
        Gui.addWorkbench(MachinaQWorkbench())
        FreeCAD.Console.PrintLog("MachinaQCAM: MachinaQ workbench registered.\n")
    except Exception as exc:
        FreeCAD.Console.PrintError(f"MachinaQCAM: Gui.addWorkbench() failed: {exc}\n")

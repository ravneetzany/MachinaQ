"""MachinaQCAM addon entry point.

Registers MachinaQ as its own **standalone FreeCAD workbench** — not
injected into CAM's toolbar. See design.md decision 2 for the full,
multi-round debugging history behind every choice in this file; the short
version of the most recent findings:

- FreeCAD executes `InitGui.py` via `exec()` without setting `__file__` —
  don't use it; resolve the addon's own directory via `FreeCAD.getUserAppDataDir()`
  / `FreeCAD.ConfigGet("AppHomePath")` instead.
- That `exec()` also uses *separate* globals/locals dicts: a plain
  top-level `NAME = value` or `def name(): ...` lands in the locals dict,
  visible to other code still running at that same top level, but
  **invisible from inside any nested class/function body** — including
  from inside *another* top-level-defined function calling it, since each
  function's `__globals__` is bound to the real globals dict, not the
  locals dict it was defined alongside. Confirmed by direct simulation of
  this exact split-dict `exec()` pattern (not just inferred from the
  traceback): a version with `_resolve_addon_dir()` as its own top-level
  `def`, called from inside `_init_machinaq_addon()`, raised `NameError:
  name '_resolve_addon_dir' is not defined` — the identical class of bug
  as the `_ICON_PATH`-in-class-body failure one round earlier, just one
  level deeper.

The fix: put genuinely everything — every helper, the class, the
registration call — inside **one** top-level function, with no other
top-level `def`s or classes at all, and call it once at the bottom. A
function's own nested scopes follow normal, well-defined Python closure
rules regardless of how the *enclosing* module was executed; this
sidesteps needing to know FreeCAD's exact `exec()` signature entirely, and
was verified by directly simulating FreeCAD's suspected
`exec(code, globals_dict, separate_locals_dict)` pattern against this file
before shipping it.
"""

import FreeCAD


def _init_machinaq_addon() -> None:
    # Re-imported here, not just at module top level: under the split-dict
    # exec() behavior described above, even a top-level `import` statement
    # lands in the locals dict and is invisible from inside this nested
    # function's own scope — only names bound *within* this function body
    # (including via a local `import` statement here) are reliably visible
    # to it and to anything nested inside it (the class below).
    import os
    import sys
    import FreeCAD
    import FreeCADGui as Gui

    def resolve_addon_dir() -> str:
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
        for entry in sys.path:
            if entry.rstrip("/").endswith("MachinaQCAM") and os.path.isfile(os.path.join(entry, "commands.py")):
                return entry
        raise RuntimeError("MachinaQCAM: could not determine the addon's own directory")

    addon_dir = resolve_addon_dir()
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)

    icon_path = os.path.join(addon_dir, "Resources", "icons", "machinaq_classify.svg")
    command_name = "MachinaQ_ClassifyFeature"

    workbench_base = getattr(Gui, "Workbench", None) or globals().get("Workbench")
    if workbench_base is None:
        FreeCAD.Console.PrintError(
            "MachinaQCAM: could not resolve a Workbench base class "
            "(neither Gui.Workbench nor a bare 'Workbench' global is available) "
            "— the MachinaQ workbench was not registered.\n"
        )
        return

    class MachinaQWorkbench(workbench_base):
        MenuText = "MachinaQ"
        ToolTip = "MachinaQ feature/operation classification"
        Icon = icon_path

        def Initialize(self):
            import commands
            commands.register()
            self.appendToolbar("MachinaQ", [command_name])
            self.appendMenu("MachinaQ", [command_name])

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


try:
    _init_machinaq_addon()
except Exception as exc:
    FreeCAD.Console.PrintError(f"MachinaQCAM: addon initialization failed: {exc}\n")

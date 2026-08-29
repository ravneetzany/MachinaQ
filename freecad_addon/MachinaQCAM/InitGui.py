"""MachinaQCAM addon entry point.

Registers the `MachinaQ_ClassifyFeature` command and adds it to the CAM
workbench's toolbar — not a new standalone workbench.

**Debugging history (three real-install rounds, each fixing a distinct bug
without fully resolving the toolbar's appearance):**
1. Originally monkey-patched `CAMWorkbench.Initialize` (runs once, on first
   activation) — failed whenever CAM was already active at startup, since
   `Initialize()` had already run before the patch installed. No error.
2. Replaced with `Gui.getMainWindow().workbenchActivated`, a signal that
   fires on every switch, polled into existence via a repeating `QTimer`
   (the same pattern FreeCAD's own bundled `Tux/PersistentToolbarsGui.py`
   addon uses) — command registration still failed, traced to
   `Gui.getMainWindow()` being called and cached once at module-import
   time, too early in startup, with the resulting `None` never re-fetched.
3. Fixed the caching bug (fetch fresh each poll) and anchored the timer's
   Python reference to the `FreeCAD` module to rule out premature garbage
   collection — command registration now reliably succeeds, but the
   *toolbar-append step* still didn't fire automatically, even though
   calling `Gui.activeWorkbench().appendToolbar(...)` manually in the
   console always worked immediately, no error, every time.

That last fact — every individual piece works when called directly, but
the deferred/polled auto-wiring silently doesn't — is the actual signal
that whatever is unreliable is the *deferral mechanism itself* in this
environment, not any of the pieces it calls. So this version drops
deferral as the primary path: it tries the toolbar-append **immediately,
synchronously**, at `InitGui.py` load time (this is what actually needs to
work for the reported case — CAM already active at startup) and layers
multiple independent, low-complexity fallbacks on top for the case where
that's too early, rather than one polling mechanism the addon depends on
entirely.
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


def _add_toolbar_if_cam_active() -> bool:
    global _toolbar_added
    if _toolbar_added:
        return True
    try:
        active = Gui.activeWorkbench()
    except Exception:
        return False
    if active is None or active.__class__.__name__ != _CAM_WORKBENCH_CLASS_NAME:
        return False
    try:
        active.appendToolbar(_TOOLBAR_NAME, [_COMMAND_NAME])
        _toolbar_added = True
        return True
    except Exception as exc:
        FreeCAD.Console.PrintWarning(f"MachinaQCAM: could not append toolbar: {exc}\n")
        return False


def _on_workbench_activated(_name=None) -> None:
    _add_toolbar_if_cam_active()


def _try_connect_workbench_signal() -> None:
    """Best-effort: connect to future workbench switches, for the case
    where CAM isn't the active workbench yet. Failure here is non-fatal —
    the retry schedule below still covers CAM becoming active later via
    repeated direct checks."""
    try:
        mw = Gui.getMainWindow()
        if mw is not None:
            mw.workbenchActivated.connect(_on_workbench_activated)
    except Exception:
        pass


def _retry(_step: int = 0) -> None:
    """One-shot retries (not a repeating timer — see module docstring for
    why) at increasing delays, each independently attempting the toolbar
    append. `QtCore.QTimer.singleShot` manages its own timer lifetime
    internally, so this doesn't depend on us keeping any timer object
    referenced/alive."""
    if _add_toolbar_if_cam_active():
        return
    if _step == 0:
        _try_connect_workbench_signal()
    delays = (250, 1000, 3000, 7000, 15000)
    if _step < len(delays):
        QtCore.QTimer.singleShot(delays[_step], lambda: _retry(_step + 1))


def _install_initialize_fallback() -> None:
    """Extra, low-cost fallback layer: also wrap CAMWorkbench.Initialize
    (approach #1 above) so a fresh CAM activation that happens *after* all
    the retries above have exhausted still gets the toolbar appended.
    Redundant with the mechanisms above in the common case, but each of
    these three mechanisms is independent and cheap, so there's no reason
    to rely on just one after three rounds of a single mechanism turning
    out to be unreliable here."""
    try:
        cam_workbench = Gui.getWorkbench(_CAM_WORKBENCH_CLASS_NAME)
    except Exception:
        return
    original_initialize = cam_workbench.__class__.Initialize
    if getattr(original_initialize, "_machinaq_wrapped", False):
        return

    def wrapped_initialize(self):
        original_initialize(self)
        _add_toolbar_if_cam_active()

    wrapped_initialize._machinaq_wrapped = True
    cam_workbench.__class__.Initialize = wrapped_initialize


commands.register()
_add_toolbar_if_cam_active()  # immediate, synchronous attempt first
_install_initialize_fallback()
_retry()

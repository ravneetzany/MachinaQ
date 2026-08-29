"""Read-only FreeCAD task panel showing a MachinaQ classification report.

Per design.md decision 5: report-only, no editing, no "apply to CAM
operation" actions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from PySide import QtGui, QtCore  # noqa: F401  (FreeCAD bundles PySide)


class ClassificationTaskPanel:
    """FreeCAD task-dialog protocol: form, accept(), reject(), getStandardButtons()."""

    def __init__(self, report: Dict[str, Any], correlations: List[Dict[str, Any]]) -> None:
        self.report = report
        self.correlations = correlations
        self.form = self._build_form()

    def _build_form(self) -> QtGui.QWidget:
        widget = QtGui.QWidget()
        widget.setWindowTitle("MachinaQ Classification")
        layout = QtGui.QVBoxLayout(widget)

        summary = self.report.get("operations_summary", {})
        summary_box = QtGui.QGroupBox("Part-level operation summary")
        summary_layout = QtGui.QFormLayout(summary_box)
        summary_layout.addRow("Primary process:", QtGui.QLabel(str(summary.get("primary_process", "-"))))
        secondary = summary.get("secondary_processes") or []
        summary_layout.addRow("Secondary processes:", QtGui.QLabel(", ".join(secondary) or "(none)"))
        rationale_label = QtGui.QLabel(str(summary.get("rationale", "")))
        rationale_label.setWordWrap(True)
        summary_layout.addRow("Rationale:", rationale_label)
        layout.addWidget(summary_box)

        features = self.report.get("features", [])
        table = QtGui.QTableWidget(len(features), 3)
        table.setHorizontalHeaderLabels(["Feature type", "Operation", "Rationale"])
        for row, feature in enumerate(features):
            table.setItem(row, 0, QtGui.QTableWidgetItem(str(feature.get("feature_type", ""))))
            table.setItem(row, 1, QtGui.QTableWidgetItem(str(feature.get("operation", ""))))
            table.setItem(row, 2, QtGui.QTableWidgetItem(str(feature.get("operation_rationale", ""))))
        table.resizeColumnsToContents()
        layout.addWidget(QtGui.QLabel("Features:"))
        layout.addWidget(table)

        if self.correlations:
            layout.addWidget(self._build_correlation_section())

        return widget

    def _build_correlation_section(self) -> QtGui.QGroupBox:
        box = QtGui.QGroupBox("Selected-face correlation (approximate match — not exact)")
        box_layout = QtGui.QVBoxLayout(box)
        note = QtGui.QLabel(
            "Nearest classified feature by geometric position. This is an "
            "approximation, not a guaranteed exact correspondence to your "
            "selection — see MachinaQCAM's README."
        )
        note.setWordWrap(True)
        box_layout.addWidget(note)

        for i, correlation in enumerate(self.correlations):
            match = correlation.get("match")
            if match is None:
                line = QtGui.QLabel(f"Face {i + 1}: no close match found")
            else:
                line = QtGui.QLabel(
                    f"Face {i + 1}: nearest feature is a {match.get('type', '?')} primitive "
                    f"(face_id={match.get('face_id', '?')}, distance={match.get('distance', 0.0):.2f})"
                )
            line.setWordWrap(True)
            box_layout.addWidget(line)

        return box

    def accept(self) -> bool:
        return True

    def reject(self) -> bool:
        return True

    def getStandardButtons(self):  # noqa: N802 (FreeCAD's naming convention)
        return QtGui.QDialogButtonBox.Close

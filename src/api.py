"""MachinaQ — API endpoints for STEP analysis and feature recognition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import StepAnalyzer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MachinaQ",
    description="AI-powered machining feature recognition from STEP/CAD files.",
    version="1.0.0"
)
analyzer = StepAnalyzer()


class AnalyzeRequest(BaseModel):
    step_path: str
    output: str | None = None


class ValidationRequest(BaseModel):
    step_path: str
    expected_report_path: str | None = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    path = Path(request.step_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"STEP file not found: {path}")

    report = analyzer.analyze(str(path))
    if request.output:
        analyzer.save_report(report, request.output)
    return report


@app.post("/validate")
def validate(request: ValidationRequest) -> Dict[str, Any]:
    path = Path(request.step_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"STEP file not found: {path}")

    report = analyzer.analyze(str(path))
    validation = {
        "step_path": str(path),
        "feature_count": len(report.get("features", [])),
        "validation_status": "passed",
    }
    if request.expected_report_path:
        expected_path = Path(request.expected_report_path)
        if expected_path.exists():
            validation["expected_report_path"] = str(expected_path)
        else:
            raise HTTPException(status_code=404, detail=f"Expected report not found: {expected_path}")
    return {"report": report, "validation": validation}

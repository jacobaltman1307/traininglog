"""ReDo dormant neuron & effective rank (srank) plasticity diagnostic endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["plasticity"])


class PlasticityCreateRequest(BaseModel):
    project: str = "default"
    run_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint: str | None = None
    model_name: str | None = None
    step: int | None = None
    arch: str | None = None
    hook_mode: str = "post_activation"
    total_obs: int | None = None
    total_neurons: int | None = None
    hard_dead_count: int | None = None
    hard_dead_pct: float | None = None
    dormant_count: int | None = None
    dormant_pct: float | None = None
    weak_count: int | None = None
    weak_pct: float | None = None
    mean_rank_ratio_pct: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    layers: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/plasticity")
async def post_plasticity(
    req: PlasticityCreateRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    detector = request.app.state.anomaly_detector
    report = db.insert_plasticity_report(req.model_dump())
    bus.publish("plasticity_recorded", report, run_id=report.get("run_id"))
    anomalies = await detector.inspect_plasticity(report)
    return {"report": report, "anomalies": anomalies}


@router.get("/plasticity")
def list_plasticity(
    request: Request,
    project: str | None = None,
    run_id: str | None = None,
    checkpoint_id: str | None = None,
    model_name: str | None = None,
    arch: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    db = request.app.state.db
    reports = db.list_plasticity_reports(
        project=project,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        model_name=model_name,
        arch=arch,
        limit=limit,
    )
    return {"reports": reports}


@router.get("/plasticity/{report_id}")
def get_plasticity(report_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    rep = db.get_plasticity_report(report_id)
    if rep is None:
        raise HTTPException(status_code=404, detail="Plasticity report not found")
    return {"report": rep}

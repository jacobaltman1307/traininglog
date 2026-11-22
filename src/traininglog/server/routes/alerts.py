"""Anomaly and alert rule endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["alerts"])

ALLOWED_CHANNELS = {"ntfy", "discord", "pushover", "telegram", "webhook"}


class AlertRuleCreateRequest(BaseModel):
    project_id: str | None = None
    name: str
    channel: str
    target: dict[str, Any]
    events: list[str] = Field(default_factory=lambda: ["crashed", "nan_inf", "stalled"])
    thresholds: dict[str, Any] = Field(default_factory=dict)
    cooldown_sec: int = 300
    enabled: bool = True


class AlertRulePatchRequest(BaseModel):
    name: str | None = None
    channel: str | None = None
    target: dict[str, Any] | None = None
    events: list[str] | None = None
    thresholds: dict[str, Any] | None = None
    cooldown_sec: int | None = None
    enabled: bool | None = None


@router.get("/anomalies")
def list_anomalies(
    request: Request,
    run_id: str | None = None,
    project_id: str | None = None,
    resolved: int | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    db = request.app.state.db
    items = db.list_anomalies(
        run_id=run_id, project_id=project_id, resolved=resolved, limit=limit
    )
    return {"anomalies": items}


@router.post("/anomalies/{anomaly_id}/resolve")
def resolve_anomaly(anomaly_id: int, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    item = db.resolve_anomaly(anomaly_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    bus.publish("anomaly_resolved", item, run_id=item["run_id"])
    return {"anomaly": item}


@router.get("/alerts/rules")
def list_alert_rules(request: Request) -> dict[str, Any]:
    db = request.app.state.db
    return {"rules": db.list_alert_rules()}


@router.post("/alerts/rules")
def create_alert_rule(req: AlertRuleCreateRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    if req.channel not in ALLOWED_CHANNELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported channel '{req.channel}'. Allowed: {sorted(ALLOWED_CHANNELS)}",
        )
    rule = db.create_alert_rule(req.model_dump())
    return {"rule": rule}


@router.patch("/alerts/rules/{rule_id}")
def patch_alert_rule(
    rule_id: str, req: AlertRulePatchRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    updates = req.model_dump(exclude_unset=True)
    if "channel" in updates and updates["channel"] not in ALLOWED_CHANNELS:
        raise HTTPException(status_code=400, detail="Unsupported channel")
    rule = db.update_alert_rule(rule_id, updates)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"rule": rule}


@router.delete("/alerts/rules/{rule_id}")
def delete_alert_rule(rule_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    if not db.delete_alert_rule(rule_id):
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"deleted": True, "rule_id": rule_id}


@router.post("/alerts/rules/{rule_id}/test")
async def test_alert_rule(rule_id: str, request: Request) -> dict[str, Any]:
    dispatcher = request.app.state.alert_dispatcher
    ev = await dispatcher.send_test(rule_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"event": ev}


@router.get("/alerts/events")
def list_alert_events(
    request: Request, limit: int = Query(100, ge=1, le=500)
) -> dict[str, Any]:
    db = request.app.state.db
    return {"events": db.list_alert_events(limit=limit)}

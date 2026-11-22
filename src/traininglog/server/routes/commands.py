"""Cooperative remote control command endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["commands"])

ALLOWED_COMMAND_TYPES = {"stop", "checkpoint", "eval", "run_redo"}


class CreateCommandRequest(BaseModel):
    cmd_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AckCommandRequest(BaseModel):
    status: str = "acked"
    result_message: str | None = None


@router.post("/runs/{run_id}/commands")
def create_command(
    run_id: str, req: CreateCommandRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    if req.cmd_type not in ALLOWED_COMMAND_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid command type '{req.cmd_type}'. Allowed: {sorted(ALLOWED_COMMAND_TYPES)}",
        )
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    cmd = db.create_command(run_id, req.cmd_type, req.payload)
    bus.publish("command_updated", cmd, run_id=run_id)
    return {"command": cmd}


@router.get("/runs/{run_id}/commands")
def list_commands(
    run_id: str,
    request: Request,
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    db = request.app.state.db
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    commands = db.list_commands(run_id, status=status, limit=limit)
    return {"commands": commands}


@router.post("/runs/{run_id}/commands/{cmd_id}/ack")
def ack_command(
    run_id: str, cmd_id: str, req: AckCommandRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    if req.status not in ("acked", "completed", "failed"):
        raise HTTPException(status_code=400, detail="Invalid command ack status")
    cmd = db.ack_command(
        run_id, cmd_id, status=req.status, result_message=req.result_message
    )
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"Command '{cmd_id}' not found")
    bus.publish("command_updated", cmd, run_id=run_id)
    return {"command": cmd}

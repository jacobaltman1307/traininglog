"""Checkpoint & model lineage registry endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["checkpoints"])


class CheckpointCreateRequest(BaseModel):
    id: str | None = None
    project: str = "default"
    run_id: str | None = None
    name: str
    step: int = 0
    parent_id: str | None = None
    parent: str | None = None
    arch: str | None = None
    obs_dim: int | None = None
    manifest_hash: str | None = None
    is_bot: bool = False
    storage_mode: str = "path_ref"
    path_or_blob: str = ""
    host: str = ""
    sha256: str | None = None
    size_bytes: int | None = None
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class CheckpointPatchRequest(BaseModel):
    tags: list[str] | None = None
    pinned: bool | None = None
    metrics: dict[str, Any] | None = None
    parent_id: str | None = None
    arch: str | None = None
    manifest_hash: str | None = None


@router.post("/checkpoints")
def create_or_update_checkpoint(
    req: CheckpointCreateRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    ckpt = db.upsert_checkpoint(req.model_dump())
    bus.publish("checkpoint_updated", ckpt, run_id=ckpt.get("run_id"))
    return {"checkpoint": ckpt}


@router.post("/checkpoints/upload")
def upload_checkpoint(
    request: Request,
    file: UploadFile = File(...),
    project: str = Form("default"),
    name: str = Form(...),
    step: int = Form(0),
    run_id: str | None = Form(None),
    parent_id: str | None = Form(None),
    arch: str | None = Form(None),
    obs_dim: int | None = Form(None),
    manifest_hash: str | None = Form(None),
    tags_json: str = Form("[]"),
    pinned: bool = Form(False),
    metrics_json: str = Form("{}"),
) -> dict[str, Any]:
    db = request.app.state.db
    blob_store = request.app.state.blob_store
    bus = request.app.state.event_bus

    sha256, size_bytes = blob_store.save_stream("checkpoints", file.file)
    tags = json.loads(tags_json or "[]")
    metrics = json.loads(metrics_json or "{}")
    ckpt = db.upsert_checkpoint(
        {
            "project": project,
            "run_id": run_id,
            "name": name,
            "step": step,
            "parent_id": parent_id,
            "arch": arch,
            "obs_dim": obs_dim,
            "manifest_hash": manifest_hash,
            "storage_mode": "uploaded",
            "path_or_blob": file.filename or f"{name}.pt",
            "sha256": sha256,
            "size_bytes": size_bytes,
            "tags": tags,
            "pinned": pinned,
            "metrics": metrics,
        }
    )
    bus.publish("checkpoint_updated", ckpt, run_id=ckpt.get("run_id"))
    return {"checkpoint": ckpt}


@router.get("/checkpoints")
def list_checkpoints(
    request: Request,
    project: str | None = None,
    run_id: str | None = None,
    arch: str | None = None,
    manifest_hash: str | None = None,
    tag: str | None = None,
    pinned: bool | None = None,
    include_bots: bool = Query(True),
    saved_only: bool = Query(False),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, Any]:
    db = request.app.state.db
    items = db.list_checkpoints(
        project=project,
        run_id=run_id,
        arch=arch,
        manifest_hash=manifest_hash,
        tag=tag,
        pinned=pinned,
        include_bots=include_bots,
        saved_only=saved_only,
        limit=limit,
    )
    return {"checkpoints": items}


@router.get("/checkpoints/lineage")
def get_lineage(request: Request, project: str | None = None) -> dict[str, Any]:
    db = request.app.state.db
    return db.get_checkpoint_lineage(project=project)


@router.get("/checkpoints/{ckpt_id}")
def get_checkpoint(ckpt_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    ckpt = db.get_checkpoint(ckpt_id)
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    history = db.get_rating_history(agent_ids=[ckpt_id]).get(ckpt_id, [])
    plasticity = db.list_plasticity_reports(checkpoint_id=ckpt_id, limit=25)
    children = [
        c
        for c in db.list_checkpoints(project=ckpt["project_id"], limit=500)
        if c.get("parent_id") == ckpt_id
    ]
    return {
        "checkpoint": ckpt,
        "children": children,
        "rating_history": history,
        "plasticity_reports": plasticity,
    }


@router.patch("/checkpoints/{ckpt_id}")
def patch_checkpoint(
    ckpt_id: str, req: CheckpointPatchRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    ckpt = db.update_checkpoint(ckpt_id, req.model_dump(exclude_unset=True))
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    bus.publish("checkpoint_updated", ckpt, run_id=ckpt.get("run_id"))
    return {"checkpoint": ckpt}


@router.delete("/checkpoints/{ckpt_id}")
def delete_checkpoint(ckpt_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    ckpt = db.delete_checkpoint(ckpt_id)
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"deleted": True, "checkpoint_id": ckpt_id}


@router.get("/checkpoints/{ckpt_id}/download")
def download_checkpoint(ckpt_id: str, request: Request):
    db = request.app.state.db
    blob_store = request.app.state.blob_store
    ckpt = db.get_checkpoint(ckpt_id)
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    if ckpt.get("storage_mode") != "uploaded" or not ckpt.get("sha256"):
        raise HTTPException(
            status_code=400,
            detail="Checkpoint is a path reference (not uploaded to blob store)",
        )
    blob_path = blob_store.get_path("checkpoints", ckpt["sha256"])
    if blob_path is None:
        raise HTTPException(status_code=404, detail="Checkpoint blob missing on disk")
    fname = ckpt.get("path_or_blob") or f"{ckpt['name']}.pt"
    return FileResponse(
        blob_path,
        filename=fname.split("/")[-1],
        media_type="application/octet-stream",
    )

"""Query, comparison, metadata, console, histogram, and export endpoints."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import sanitize_json_value

router = APIRouter(prefix="/api", tags=["runs"])


class RunPatchRequest(BaseModel):
    name: str | None = None
    group_name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    state: str | None = None


class CompareSeriesRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    max_points: int = 800
    x_axis: str = "step"
    min_step: int | None = None
    max_step: int | None = None


class CompareConfigDiffRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


@router.get("/projects")
def list_projects(request: Request) -> dict[str, Any]:
    db = request.app.state.db
    return {"projects": db.list_projects()}


@router.get("/runs")
def list_runs(
    request: Request,
    project: str | None = None,
    state: str | None = None,
    group_name: str | None = None,
    arch: str | None = None,
    tag: str | None = None,
    search: str | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    db = request.app.state.db
    runs = db.list_runs(
        project=project,
        state=state,
        group_name=group_name,
        arch=arch,
        tag=tag,
        search=search,
        limit=limit,
    )
    return {"runs": runs}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    series_store = request.app.state.series_store
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    metrics = series_store.list_metrics(run_id)
    histograms = db.list_histogram_names(run_id)
    anomalies = db.list_anomalies(run_id=run_id, resolved=0, limit=50)
    commands = db.list_commands(run_id, limit=25)
    return {
        "run": run,
        "metrics": metrics,
        "histograms": histograms,
        "open_anomalies": anomalies,
        "commands": commands,
    }


@router.patch("/runs/{run_id}")
def patch_run(run_id: str, req: RunPatchRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    updates = req.model_dump(exclude_unset=True)
    run = db.update_run_metadata(run_id, updates)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    bus.publish("run_updated", run, run_id=run_id)
    return {"run": run}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    series_store = request.app.state.series_store
    bus = request.app.state.event_bus
    deleted = db.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    series_store.delete_run(run_id)
    bus.publish("run_deleted", {"id": run_id}, run_id=run_id)
    return {"deleted": True, "run_id": run_id}


@router.get("/runs/{run_id}/series")
def get_run_series(
    run_id: str,
    request: Request,
    metrics: str = Query(..., description="Comma-separated metric names"),
    max_points: int = Query(800, ge=8, le=10000),
    x_axis: str = Query("step"),
    min_step: int | None = None,
    max_step: int | None = None,
) -> dict[str, Any]:
    db = request.app.state.db
    series_store = request.app.state.series_store
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    out: dict[str, Any] = {}
    for m in metric_list:
        out[m] = series_store.query_series(
            run_id,
            m,
            max_points=max_points,
            x_axis=x_axis,
            run_started_ms=int(run["started_at"]),
            min_step=min_step,
            max_step=max_step,
        )
    return {"run_id": run_id, "x_axis": x_axis, "series": out}


@router.post("/compare/series")
def compare_series(req: CompareSeriesRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    series_store = request.app.state.series_store

    runs_meta: dict[str, Any] = {}
    for rid in req.run_ids:
        r = db.get_run(rid)
        if r is not None:
            runs_meta[rid] = {
                "id": r["id"],
                "name": r["name"],
                "arch": r["arch"],
                "state": r["state"],
                "started_at": r["started_at"],
            }

    metric_names = list(req.metrics)
    if not metric_names and runs_meta:
        discovered: set[str] = set()
        for rid in runs_meta:
            for item in series_store.list_metrics(rid):
                discovered.add(item["name"])
        metric_names = sorted(discovered)

    series_out: dict[str, dict[str, Any]] = {}
    for m in metric_names:
        per_run: dict[str, Any] = {}
        for rid, rmeta in runs_meta.items():
            res = series_store.query_series(
                rid,
                m,
                max_points=req.max_points,
                x_axis=req.x_axis,
                run_started_ms=int(rmeta["started_at"]),
                min_step=req.min_step,
                max_step=req.max_step,
            )
            if res["total_points"] > 0:
                per_run[rid] = res
        series_out[m] = per_run

    return {
        "x_axis": req.x_axis,
        "runs": runs_meta,
        "metrics": metric_names,
        "series": series_out,
    }


@router.post("/compare/config-diff")
def compare_config_diff(
    req: CompareConfigDiffRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    runs: list[dict[str, Any]] = []
    for rid in req.run_ids:
        r = db.get_run(rid)
        if r is not None:
            runs.append(r)

    all_keys: set[str] = set()
    for r in runs:
        all_keys.update((r.get("config") or {}).keys())

    rows: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        values: dict[str, Any] = {}
        serialized: set[str] = set()
        for r in runs:
            cfg = r.get("config") or {}
            has_key = key in cfg
            val = cfg.get(key) if has_key else None
            values[r["id"]] = sanitize_json_value(val) if has_key else "__MISSING__"
            serialized.add(json.dumps(values[r["id"]], sort_keys=True))
        rows.append(
            {
                "key": key,
                "values": {
                    rid: (None if v == "__MISSING__" else v)
                    for rid, v in values.items()
                },
                "present": {
                    rid: (v != "__MISSING__") for rid, v in values.items()
                },
                "differs": len(serialized) > 1,
            }
        )

    return {
        "runs": [
            {
                "id": r["id"],
                "name": r["name"],
                "arch": r["arch"],
                "state": r["state"],
                "group_name": r["group_name"],
            }
            for r in runs
        ],
        "rows": rows,
    }


@router.get("/runs/{run_id}/histograms")
def get_run_histograms(
    run_id: str,
    request: Request,
    name: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    db = request.app.state.db
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if name:
        items = db.get_histograms(run_id, name, limit=limit)
        return {"run_id": run_id, "name": name, "snapshots": items}
    names = db.list_histogram_names(run_id)
    return {"run_id": run_id, "histograms": names}


@router.get("/runs/{run_id}/console")
def get_run_console(
    run_id: str,
    request: Request,
    after_id: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    stream: str | None = None,
) -> dict[str, Any]:
    db = request.app.state.db
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    lines = db.get_console_logs(run_id, after_id=after_id, limit=limit, stream=stream)
    return {"run_id": run_id, "lines": lines}


@router.get("/runs/{run_id}/export")
def export_run(
    run_id: str,
    request: Request,
    format: str = Query("csv", pattern="^(csv|json)$"),
) -> Response:
    db = request.app.state.db
    series_store = request.app.state.series_store
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    rows = series_store.export_run_rows(run_id)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run["name"])

    if format == "json":
        payload = json.dumps(
            {"run": sanitize_json_value(run), "scalars": rows},
            indent=2,
            allow_nan=False,
        )
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}.json"'
            },
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["step", "wall_ms", "metric", "value"])
    for r in rows:
        writer.writerow([r["step"], r["wall_ms"], r["metric"], "" if r["value"] is None else r["value"]])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.csv"'
        },
    )

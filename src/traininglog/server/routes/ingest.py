"""Ingestion endpoints for runs, batched scalars/histograms/console/heartbeats, and completion."""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..db import now_ms, sanitize_json_value
from ..series import _parse_float

router = APIRouter(prefix="/api", tags=["ingest"])


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    project: str = "default"
    name: str | None = None
    group_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    arch: str | None = None
    obs_dim: int | None = None
    manifest_hash: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    git: dict[str, Any] = Field(default_factory=dict)
    system_info: dict[str, Any] = Field(default_factory=dict)
    cli_args: str = ""
    host: str = ""
    parent_checkpoint_id: str | None = None
    total_timesteps: int | None = None
    started_at: int | None = None


class EventBatchRequest(BaseModel):
    seq: int
    events: list[dict[str, Any]] = Field(default_factory=list)


class RunFinishRequest(BaseModel):
    state: str = "finished"
    summary: dict[str, Any] = Field(default_factory=dict)
    finished_at: int | None = None
    error_message: str | None = None


@router.post("/runs")
async def create_or_resume_run(req: RunCreateRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    run, created = db.create_or_resume_run(req.model_dump())
    bus.publish("run_created" if created else "run_updated", run, run_id=run["id"])
    return {"run": run, "created": created}


@router.post("/runs/{run_id}/events")
async def ingest_events(
    run_id: str, req: EventBatchRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    series_store = request.app.state.series_store
    bus = request.app.state.event_bus
    detector = request.app.state.anomaly_detector

    existing = db.get_run(run_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if not req.events:
        if not db.check_and_mark_batch(run_id, req.seq):
            return {"status": "duplicate", "seq": req.seq, "accepted": 0}
        return {"status": "ok", "seq": req.seq, "accepted": 0, "anomalies": 0}

    all_have_event_seq = all("event_seq" in ev and ev["event_seq"] is not None for ev in req.events)
    if all_have_event_seq:
        event_seqs = [int(ev["event_seq"]) for ev in req.events]
        new_seqs = db.filter_and_mark_event_seqs(run_id, event_seqs)
        if not new_seqs:
            return {"status": "duplicate", "seq": req.seq, "accepted": 0}
        events_to_process = [ev for ev in req.events if int(ev["event_seq"]) in new_seqs]
    else:
        if not db.check_and_mark_batch(run_id, req.seq):
            return {"status": "duplicate", "seq": req.seq, "accepted": 0}
        events_to_process = req.events

    points_raw: dict[str, list[tuple[int, int, Any]]] = {}
    points_parsed: dict[str, list[tuple[int, int, float]]] = {}
    histograms: list[dict[str, Any]] = []
    console_items: list[dict[str, Any]] = []
    summary_updates: dict[str, Any] = {}
    config_updates: dict[str, Any] = {}
    gpu_updates: dict[str, Any] = {}
    max_step: int | None = None
    latest_sps: float | None = None
    heartbeat_ms: int | None = None

    for ev in events_to_process:
        etype = ev.get("type")
        if etype == "scalars":
            step = int(ev.get("step", 0))
            wall_ms = int(ev.get("wall_ms") or now_ms())
            if max_step is None or step > max_step:
                max_step = step
            heartbeat_ms = max(heartbeat_ms or 0, wall_ms)
            metrics = ev.get("metrics") or {}
            for mname, raw_v in metrics.items():
                if raw_v is None:
                    continue
                try:
                    val = _parse_float(raw_v)
                except (TypeError, ValueError):
                    continue
                points_raw.setdefault(mname, []).append((step, wall_ms, raw_v))
                points_parsed.setdefault(mname, []).append((step, wall_ms, val))
                if math.isfinite(val):
                    summary_updates[mname] = float(val)
                    low = mname.lower()
                    if low in ("charts/sps", "perf/sps", "sps"):
                        latest_sps = float(val)
                    if low.startswith("sys/gpu_"):
                        gpu_updates[low.removeprefix("sys/")] = float(val)
                else:
                    summary_updates[mname] = None

        elif etype == "histogram":
            histograms.append(ev)
            step = ev.get("step")
            if step is not None:
                s_int = int(step)
                if max_step is None or s_int > max_step:
                    max_step = s_int

        elif etype == "console":
            console_items.append(ev)

        elif etype == "summary":
            vals = ev.get("values") or {}
            summary_updates.update(sanitize_json_value(vals))

        elif etype == "config":
            vals = ev.get("values") or {}
            config_updates.update(sanitize_json_value(vals))

        elif etype == "heartbeat":
            ts = int(ev.get("ts_ms") or now_ms())
            heartbeat_ms = max(heartbeat_ms or 0, ts)
            if isinstance(ev.get("gpu"), dict):
                gpu_updates.update(sanitize_json_value(ev["gpu"]))

    if points_raw:
        series_store.append_points(run_id, points_raw)
    if histograms:
        db.insert_histograms(run_id, histograms)
    inserted_logs = db.insert_console_logs(run_id, console_items)

    updated_run, revived = db.apply_run_ingest_state(
        run_id,
        max_step=max_step,
        current_sps=latest_sps,
        summary_updates=summary_updates or None,
        config_updates=config_updates or None,
        gpu_updates=gpu_updates or None,
        heartbeat_ms=heartbeat_ms or now_ms(),
    )
    assert updated_run is not None

    if revived:
        bus.publish("run_updated", updated_run, run_id=run_id)

    if points_raw or summary_updates or heartbeat_ms is not None:
        bus.publish(
            "run_metrics",
            {
                "run": updated_run,
                "updated_metrics": sorted(points_raw.keys()),
            },
            run_id=run_id,
        )

    if inserted_logs:
        bus.publish("console_batch", {"lines": inserted_logs}, run_id=run_id)

    anomalies = await detector.inspect_scalars(updated_run, points_parsed)

    return {
        "status": "ok",
        "seq": req.seq,
        "accepted": len(events_to_process),
        "anomalies": len(anomalies),
    }


@router.post("/runs/{run_id}/finish")
async def finish_run(
    run_id: str, req: RunFinishRequest, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    detector = request.app.state.anomaly_detector
    state = req.state if req.state in ("finished", "crashed", "killed") else "finished"

    run = db.finish_run(
        run_id,
        state=state,
        summary_updates=req.summary or None,
        finished_at=req.finished_at,
    )
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    await detector.on_run_finished(run, error_message=req.error_message)
    return {"run": run}

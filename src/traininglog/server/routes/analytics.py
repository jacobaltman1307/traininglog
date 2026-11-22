"""Card meta analytics and action/strategy distribution endpoints."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import sanitize_json_value

router = APIRouter(prefix="/api", tags=["analytics"])


class CardStatsRequest(BaseModel):
    project: str = "default"
    run_id: str | None = None
    step: int = 0
    cards: list[dict[str, Any]] | dict[str, Any] = Field(default_factory=list)
    recorded_at: int | None = None


class DistributionRequest(BaseModel):
    run_id: str
    name: str
    step: int = 0
    labels: list[str]
    probs: list[float]
    entropy: float | None = None
    recorded_at: int | None = None


@router.post("/cards/stats")
def record_card_stats(req: CardStatsRequest, request: Request) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if isinstance(req.cards, dict):
        for card_name, stats in req.cards.items():
            if isinstance(stats, dict):
                items.append({"card_name": card_name, **stats})
            else:
                items.append({"card_name": card_name, "played": int(stats or 0)})
    else:
        items = list(req.cards)

    db = request.app.state.db
    bus = request.app.state.event_bus
    result = db.insert_card_stats(
        project=req.project,
        run_id=req.run_id,
        step=req.step,
        items=items,
        recorded_at=req.recorded_at,
    )
    bus.publish("cards_updated", result, run_id=result.get("run_id"))
    return result


@router.get("/cards/meta")
def get_card_meta(
    request: Request,
    project: str | None = Query(None),
    run_id: str | None = Query(None),
    min_played: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> Response:
    cards = request.app.state.db.get_card_meta(
        project=project,
        run_id=run_id,
        min_played=min_played,
        limit=limit,
    )
    body = json.dumps({"cards": sanitize_json_value(cards)}, allow_nan=False)
    return Response(content=body, media_type="application/json")


@router.get("/cards/trends")
def get_card_trends(
    request: Request,
    project: str | None = Query(None),
    run_id: str | None = Query(None),
    cards: str | None = Query(None),
    limit_cards: int = Query(8, ge=1, le=32),
) -> Response:
    card_list = [c.strip() for c in (cards or "").split(",") if c.strip()] or None
    trends = request.app.state.db.get_card_trends(
        project=project,
        run_id=run_id,
        cards=card_list,
        limit_cards=limit_cards,
    )
    body = json.dumps({"trends": sanitize_json_value(trends)}, allow_nan=False)
    return Response(content=body, media_type="application/json")


@router.get("/cards/export")
def export_card_stats(
    request: Request,
    project: str | None = Query(None),
    run_id: str | None = Query(None),
    format: str = Query("csv"),
) -> Response:
    rows = request.app.state.db.export_card_stats(project=project, run_id=run_id)
    fmt = format.strip().lower()
    if fmt == "json":
        body = json.dumps({"rows": sanitize_json_value(rows)}, allow_nan=False, indent=2)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="card_stats.json"'},
        )

    buf = io.StringIO()
    fieldnames = [
        "project_name",
        "run_id",
        "step",
        "card_name",
        "played",
        "drawn",
        "wins_when_played",
        "games_when_played",
        "play_rate",
        "win_rate",
        "recorded_at",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="card_stats.csv"'},
    )


@router.post("/distributions")
def record_distribution(req: DistributionRequest, request: Request) -> dict[str, Any]:
    if len(req.labels) != len(req.probs):
        raise HTTPException(
            status_code=400, detail="labels and probs must have the same length"
        )
    db = request.app.state.db
    bus = request.app.state.event_bus
    try:
        saved = db.insert_distribution(
            run_id=req.run_id,
            name=req.name,
            step=req.step,
            labels=req.labels,
            probs=req.probs,
            entropy=req.entropy,
            recorded_at=req.recorded_at,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    bus.publish("cards_updated", {"run_id": req.run_id, "name": req.name, "step": req.step}, run_id=req.run_id)
    return {"distribution": saved}


@router.get("/distributions")
def get_distributions(
    request: Request,
    run_id: str = Query(...),
    name: str | None = Query(None),
    limit: int = Query(500, ge=2, le=5000),
) -> Response:
    db = request.app.state.db
    if db.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    names = db.list_distribution_names(run_id)
    target_name = name or (names[0] if names else None)
    series = (
        db.get_distribution_series(run_id=run_id, name=target_name, limit=limit)
        if target_name is not None
        else []
    )
    body = json.dumps(
        sanitize_json_value(
            {
                "run_id": run_id,
                "name": target_name,
                "names": names,
                "series": series,
            }
        ),
        allow_nan=False,
    )
    return Response(content=body, media_type="application/json")

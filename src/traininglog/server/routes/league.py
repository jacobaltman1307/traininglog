"""Self-play league, match ingestion, Glicko-2 / Elo / OpenSkill leaderboards, and H2H matrix endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["league"])


class MatchCreateRequest(BaseModel):
    project: str = "default"
    run_id: str | None = None
    source_key: str | None = None
    step: int | None = None
    mode: str | None = None
    total_games: int | None = None
    draws: int = 0
    p1_side_wins: int | None = None
    p2_side_wins: int | None = None
    avg_rounds: float = 0.0
    avg_steps: float = 0.0
    duration_sec: float = 0.0
    manifest_hash: str | None = None
    replay_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    played_at: int | None = None
    participants: list[dict[str, Any]] = Field(default_factory=list)


class DirectRatingRequest(BaseModel):
    project: str = "default"
    agent: str
    glicko_mu: float | None = None
    glicko_rd: float | None = None
    elo: float | None = None
    os_mu: float | None = None
    os_sigma: float | None = None
    step: int | None = None
    run_id: str | None = None
    recorded_at: int | None = None


@router.post("/matches")
async def post_match(req: MatchCreateRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    detector = request.app.state.anomaly_detector
    if len(req.participants) < 2:
        raise HTTPException(
            status_code=400, detail="Match requires at least 2 participants"
        )
    res = db.record_match(req.model_dump())
    if not res["duplicate"]:
        bus.publish(
            "match_recorded",
            {
                "match_id": res["match_id"],
                "project_id": res["project_id"],
                "rating_updates": res["rating_updates"],
            },
            run_id=req.run_id,
        )
        if res.get("new_best"):
            nb = res["new_best"]
            await detector.on_new_best_rating(
                res["project_id"], nb["agent"], nb["glicko_mu"], nb["glicko_rd"]
            )
    return res


@router.post("/ratings/log")
def post_direct_rating(req: DirectRatingRequest, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    rating = db.log_direct_rating(req.model_dump())
    bus.publish("match_recorded", {"rating": rating}, run_id=req.run_id)
    return {"rating": rating}


@router.get("/matches")
def list_matches(
    request: Request,
    project: str | None = None,
    agent_id: str | None = None,
    opponent_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    db = request.app.state.db
    matches = db.list_matches(
        project=project, agent_id=agent_id, opponent_id=opponent_id, limit=limit
    )
    return {"matches": matches}


@router.get("/league/leaderboard")
def get_leaderboard(
    request: Request,
    project: str | None = None,
    arch: str | None = None,
    manifest_hash: str | None = None,
    saved_only: bool = Query(False),
    include_bots: bool = Query(True),
    include_unplayed: bool = Query(False),
    search: str | None = None,
    sort_by: str = Query("glicko_mu"),
) -> dict[str, Any]:
    db = request.app.state.db
    entries = db.get_leaderboard(
        project=project,
        arch=arch,
        manifest_hash=manifest_hash,
        saved_only=saved_only,
        include_bots=include_bots,
        include_unplayed=include_unplayed,
        search=search,
        sort_by=sort_by,
    )
    return {"entries": entries}


@router.get("/league/matrix")
def get_matrix(
    request: Request,
    project: str | None = None,
    arch: str | None = None,
    manifest_hash: str | None = None,
    saved_only: bool = Query(False),
    include_bots: bool = Query(True),
    search: str | None = None,
) -> dict[str, Any]:
    db = request.app.state.db
    return db.get_h2h_matrix(
        project=project,
        arch=arch,
        manifest_hash=manifest_hash,
        saved_only=saved_only,
        include_bots=include_bots,
        search=search,
    )


@router.get("/league/seats")
def get_seats(request: Request, project: str | None = None) -> dict[str, Any]:
    db = request.app.state.db
    return db.get_seat_and_deck_stats(project=project)


@router.get("/league/rating-history")
def get_rating_history(
    request: Request,
    project: str | None = None,
    agent_ids: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    db = request.app.state.db
    ids = [x.strip() for x in agent_ids.split(",") if x.strip()] if agent_ids else None
    history = db.get_rating_history(project=project, agent_ids=ids, run_id=run_id)
    return {"history": history}


@router.post("/league/recompute")
def recompute_ratings(request: Request, project: str = Query(...)) -> dict[str, Any]:
    db = request.app.state.db
    bus = request.app.state.event_bus
    res = db.recompute_project_ratings(project)
    bus.publish("match_recorded", {"recomputed": True, "project": project})
    return res

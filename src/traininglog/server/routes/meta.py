"""Card Meta analytics, versioned catalog, and per-game telemetry ingestion endpoints."""

from __future__ import annotations

import csv
import gzip
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import sanitize_json_value
from ..meta import MetaFilter

router = APIRouter(prefix="/api/meta", tags=["meta"])


class CardCatalogUpsertRequest(BaseModel):
    project: str = "default"
    manifest_hash: str = ""
    patch_label: str = ""
    notes: str = ""
    num_lanes: int = 4
    cards: list[dict[str, Any]] | dict[str, dict[str, Any]] = Field(default_factory=list)


def _strict_json_response(payload: Any) -> Response:
    body = json.dumps(sanitize_json_value(payload), allow_nan=False)
    return Response(content=body, media_type="application/json")


def _build_filter(
    *,
    project: str | None = None,
    source: str | None = None,
    manifest_hash: str | None = None,
    mode: str = "1v1",
    exclude_bots: bool = True,
    min_rating: float | None = None,
    max_rating: float | None = None,
    tier: str | None = None,
    top_k_agents: int | None = None,
    exclude_mirrors: bool = False,
    min_games: int = 10,
    min_pair_games: int | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    faction: str | None = None,
    card_type: str | None = None,
    mana_cost_min: int | None = None,
    mana_cost_max: int | None = None,
    tag: str | None = None,
    critic_perspective: str | None = None,
) -> MetaFilter:
    return MetaFilter(
        project=project or None,
        source=source or None,
        manifest_hash=manifest_hash or None,
        mode=mode or "1v1",
        exclude_bots=exclude_bots,
        min_rating=min_rating,
        max_rating=max_rating,
        tier=tier or None,
        top_k_agents=top_k_agents,
        exclude_mirrors=exclude_mirrors,
        min_games=max(0, int(min_games)),
        min_pair_games=min_pair_games,
        since_ms=since_ms,
        until_ms=until_ms,
        faction=faction or None,
        card_type=card_type or None,
        mana_cost_min=mana_cost_min,
        mana_cost_max=mana_cost_max,
        tag=tag or None,
        critic_perspective=critic_perspective or None,
    )


@router.post("/catalog")
def upsert_catalog(req: CardCatalogUpsertRequest, request: Request) -> Response:
    if isinstance(req.cards, dict):
        cards = [
            {**(v if isinstance(v, dict) else {"display_name": str(v)}), "card_key": k}
            for k, v in req.cards.items()
        ]
    else:
        cards = list(req.cards)

    res = request.app.state.db.upsert_card_catalog(
        project=req.project,
        manifest_hash=req.manifest_hash,
        cards=cards,
        patch_label=req.patch_label,
        notes=req.notes,
        num_lanes=req.num_lanes,
    )
    request.app.state.event_bus.publish("meta_updated", res)
    return _strict_json_response(res)


@router.post("/games/batch")
async def ingest_games_batch(
    request: Request,
    project: str = Query("default"),
    source: str = Query("league"),
    manifest_hash: str = Query(""),
    value_scale: str = Query("prob"),
) -> Response:
    raw_body = await request.body()
    if "gzip" in request.headers.get("content-encoding", "").lower():
        try:
            raw_body = gzip.decompress(raw_body)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid gzip payload: {exc}") from exc

    text = raw_body.decode("utf-8", errors="replace").strip()
    if not text:
        return _strict_json_response(
            {"project": project, "inserted": 0, "duplicates": 0, "total_plays": 0}
        )

    eff_project = project
    eff_source = source
    eff_manifest = manifest_hash
    eff_match_id: str | None = None
    eff_run_id: str | None = None
    eff_scale = value_scale
    games: list[dict[str, Any]]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Support NDJSON line-by-line body as well
        games = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid NDJSON payload: {exc}") from exc
            if isinstance(obj, dict):
                games.append(obj)
    else:
        if isinstance(parsed, dict):
            eff_project = str(parsed.get("project") or project)
            eff_source = str(parsed.get("source") or source)
            eff_manifest = str(
                parsed.get("manifest_hash")
                if parsed.get("manifest_hash") is not None
                else manifest_hash
            )
            eff_match_id = parsed.get("match_id")
            eff_run_id = parsed.get("run_id")
            eff_scale = str(parsed.get("value_scale") or value_scale)
            raw_games = parsed.get("games")
            if raw_games is None:
                raw_games = [parsed] if "players" in parsed else []
            if not isinstance(raw_games, list):
                raise HTTPException(status_code=400, detail="'games' must be a list")
            games = raw_games
        elif isinstance(parsed, list):
            games = parsed
        else:
            raise HTTPException(status_code=400, detail="Expected JSON object, list, or NDJSON")

    if len(games) > 5000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 5000 games limit")

    res = request.app.state.db.insert_meta_games_batch(
        project=eff_project,
        games=games,
        default_source=eff_source,
        default_manifest_hash=eff_manifest,
        default_match_id=eff_match_id,
        default_run_id=eff_run_id,
        value_scale=eff_scale,
    )
    if res["inserted"] > 0:
        request.app.state.event_bus.publish("meta_updated", res, run_id=eff_run_id)
    return _strict_json_response(res)


@router.get("/filters")
def get_meta_filters(request: Request, project: str | None = Query(None)) -> Response:
    res = request.app.state.meta_engine.get_filters(project=project)
    return _strict_json_response(res)


@router.get("/overview")
def get_meta_overview(
    request: Request,
    project: str | None = Query(None),
    source: str | None = Query(None),
    manifest_hash: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    min_rating: float | None = Query(None),
    max_rating: float | None = Query(None),
    tier: str | None = Query(None),
    top_k_agents: int | None = Query(None, ge=1),
    exclude_mirrors: bool = Query(False),
    min_games: int = Query(10, ge=0),
    min_pair_games: int | None = Query(None, ge=1),
    since_ms: int | None = Query(None),
    until_ms: int | None = Query(None),
    faction: str | None = Query(None),
    card_type: str | None = Query(None),
    mana_cost_min: int | None = Query(None),
    mana_cost_max: int | None = Query(None),
    tag: str | None = Query(None),
    critic_perspective: str | None = Query(None),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        manifest_hash=manifest_hash,
        mode=mode,
        exclude_bots=exclude_bots,
        min_rating=min_rating,
        max_rating=max_rating,
        tier=tier,
        top_k_agents=top_k_agents,
        exclude_mirrors=exclude_mirrors,
        min_games=min_games,
        min_pair_games=min_pair_games,
        since_ms=since_ms,
        until_ms=until_ms,
        faction=faction,
        card_type=card_type,
        mana_cost_min=mana_cost_min,
        mana_cost_max=mana_cost_max,
        tag=tag,
        critic_perspective=critic_perspective,
    )
    return _strict_json_response(request.app.state.meta_engine.get_overview(flt))


@router.get("/cards/{card_key}")
def get_meta_card_detail(
    card_key: str,
    request: Request,
    project: str | None = Query(None),
    source: str | None = Query(None),
    manifest_hash: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    min_rating: float | None = Query(None),
    max_rating: float | None = Query(None),
    tier: str | None = Query(None),
    top_k_agents: int | None = Query(None, ge=1),
    exclude_mirrors: bool = Query(False),
    min_pair_games: int | None = Query(None, ge=1),
    since_ms: int | None = Query(None),
    until_ms: int | None = Query(None),
    critic_perspective: str | None = Query(None),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        manifest_hash=manifest_hash,
        mode=mode,
        exclude_bots=exclude_bots,
        min_rating=min_rating,
        max_rating=max_rating,
        tier=tier,
        top_k_agents=top_k_agents,
        exclude_mirrors=exclude_mirrors,
        min_games=0,
        min_pair_games=min_pair_games,
        since_ms=since_ms,
        until_ms=until_ms,
        critic_perspective=critic_perspective,
    )
    detail = request.app.state.meta_engine.get_card_detail(card_key, flt)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Card '{card_key}' not found in meta data")
    return _strict_json_response(detail)


@router.get("/synergies")
def get_meta_synergies(
    request: Request,
    project: str | None = Query(None),
    source: str | None = Query(None),
    manifest_hash: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    min_rating: float | None = Query(None),
    max_rating: float | None = Query(None),
    tier: str | None = Query(None),
    top_k_agents: int | None = Query(None, ge=1),
    min_pair_games: int | None = Query(None, ge=1),
    level: str = Query("played"),
    limit: int = Query(40, ge=1, le=200),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        manifest_hash=manifest_hash,
        mode=mode,
        exclude_bots=exclude_bots,
        min_rating=min_rating,
        max_rating=max_rating,
        tier=tier,
        top_k_agents=top_k_agents,
        min_games=0,
        min_pair_games=min_pair_games,
    )
    return _strict_json_response(
        request.app.state.meta_engine.get_synergies(flt, level=level, limit=limit)
    )


@router.get("/trends")
def get_meta_trends(
    request: Request,
    project: str | None = Query(None),
    source: str | None = Query(None),
    manifest_hash: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    min_rating: float | None = Query(None),
    max_rating: float | None = Query(None),
    tier: str | None = Query(None),
    top_k_agents: int | None = Query(None, ge=1),
    cards: str | None = Query(None),
    limit_cards: int = Query(8, ge=1, le=24),
    bucket: str = Query("day"),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        manifest_hash=manifest_hash,
        mode=mode,
        exclude_bots=exclude_bots,
        min_rating=min_rating,
        max_rating=max_rating,
        tier=tier,
        top_k_agents=top_k_agents,
        min_games=0,
    )
    card_list = [c.strip() for c in (cards or "").split(",") if c.strip()] or None
    return _strict_json_response(
        request.app.state.meta_engine.get_trends(
            flt, cards=card_list, limit_cards=limit_cards, bucket=bucket
        )
    )


@router.get("/patch-diff")
def get_meta_patch_diff(
    request: Request,
    manifest_a: str = Query(...),
    manifest_b: str = Query(...),
    project: str | None = Query(None),
    source: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    exclude_mirrors: bool = Query(False),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        mode=mode,
        exclude_bots=exclude_bots,
        exclude_mirrors=exclude_mirrors,
        min_games=0,
    )
    return _strict_json_response(
        request.app.state.meta_engine.get_patch_diff(flt, manifest_a, manifest_b)
    )


@router.get("/export")
def export_meta_overview(
    request: Request,
    project: str | None = Query(None),
    source: str | None = Query(None),
    manifest_hash: str | None = Query(None),
    mode: str = Query("1v1"),
    exclude_bots: bool = Query(True),
    exclude_mirrors: bool = Query(False),
    min_games: int = Query(10, ge=0),
    format: str = Query("csv"),
) -> Response:
    flt = _build_filter(
        project=project,
        source=source,
        manifest_hash=manifest_hash,
        mode=mode,
        exclude_bots=exclude_bots,
        exclude_mirrors=exclude_mirrors,
        min_games=min_games,
    )
    overview = request.app.state.meta_engine.get_overview(flt)
    cards = overview.get("cards") or []
    fmt = format.strip().lower()
    if fmt == "json":
        body = json.dumps(sanitize_json_value(overview), allow_nan=False, indent=2)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="meta_cards.json"'},
        )

    buf = io.StringIO()
    fieldnames = [
        "card_key",
        "display_name",
        "games_in_deck",
        "games_drawn",
        "games_played",
        "inclusion_rate",
        "avg_copies",
        "draw_to_play_rate",
        "oh_wr",
        "gih_wr",
        "gns_wr",
        "played_wr",
        "drawn_wpa",
        "played_lift",
        "critic_play_wpa",
        "critic_draw_wpa",
        "isolated_impact",
        "meta_war",
        "synergy_reliance",
        "top_synergy_partner",
        "top_synergy_lift",
        "favored_lane",
        "best_lane",
        "lane_concentration",
        "avg_turn_played",
        "p1_wr",
        "p2_wr",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for c in cards:
        lp = c.get("lane_preference") or {}
        ts = c.get("top_synergy") or {}
        row = {
            **{k: c.get(k) for k in fieldnames if k in c},
            "top_synergy_partner": ts.get("partner"),
            "top_synergy_lift": ts.get("lift"),
            "favored_lane": lp.get("favored_lane"),
            "best_lane": lp.get("best_lane"),
            "lane_concentration": lp.get("concentration"),
        }
        writer.writerow(row)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="meta_cards.csv"'},
    )

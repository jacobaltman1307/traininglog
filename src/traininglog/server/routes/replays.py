"""Godot ReplayRecord hub, MP4/WebM media streaming, and RL turn trace endpoints."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..db import sanitize_json_value

router = APIRouter(prefix="/api", tags=["replays"])


class ReplayPatchRequest(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    tags: list[str] | None = None


def _finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _detect_format_and_mime(
    raw_bytes: bytes,
    filename: str,
    requested_format: str | None,
    upload_content_type: str | None,
) -> tuple[str, str]:
    suffix = Path(filename or "").suffix.lower()
    req_fmt = (requested_format or "").strip().lower()

    if req_fmt and req_fmt != "auto":
        if req_fmt == "mp4":
            return "mp4", "video/mp4"
        if req_fmt == "webm":
            return "webm", "video/webm"
        if req_fmt in ("cardgame_replay_json", "json"):
            return req_fmt, "application/json"
        return req_fmt, upload_content_type or "application/octet-stream"

    # Check MP4/WebM magic headers before falling back to filename/JSON inspection.
    if len(raw_bytes) >= 12 and raw_bytes[4:8] == b"ftyp" or suffix == ".mp4":
        return "mp4", "video/mp4"
    if raw_bytes.startswith(b"\x1a\x45\xdf\xa3") or suffix == ".webm":
        return "webm", "video/webm"

    stripped = raw_bytes.lstrip()
    if stripped.startswith((b"{", b"[")) or suffix == ".json":
        return "json", "application/json"

    return "bin", upload_content_type or "application/octet-stream"


def _extract_cardgame_metadata(raw_bytes: bytes) -> dict[str, Any]:
    """Best-effort read-only extraction of Godot ReplayRecord fields without mutating raw bytes."""
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    meta: dict[str, Any] = {}
    is_cardgame = (
        isinstance(parsed.get("actions"), list)
        and any(
            k in parsed
            for k in ("version", "manifest_hash", "seed1", "seed2", "deck1", "deck2", "config")
        )
    )
    if is_cardgame:
        meta["format"] = "cardgame_replay_json"

    if parsed.get("version") is not None:
        try:
            meta["version"] = int(parsed["version"])
        except (TypeError, ValueError):
            pass
    if isinstance(parsed.get("manifest_hash"), str):
        meta["manifest_hash"] = parsed["manifest_hash"]
    for seed_key in ("seed1", "seed2"):
        if parsed.get(seed_key) is not None:
            try:
                meta[seed_key] = int(parsed[seed_key])
            except (TypeError, ValueError):
                pass
    for deck_key in ("deck1", "deck2"):
        deck_val = parsed.get(deck_key)
        if isinstance(deck_val, list):
            meta[deck_key] = [str(c) for c in deck_val]
    if isinstance(parsed.get("config"), dict):
        meta["config"] = sanitize_json_value(parsed["config"])
    actions = parsed.get("actions")
    if isinstance(actions, list):
        meta["action_count"] = len(actions)
    if parsed.get("winner") is not None:
        try:
            meta["winner"] = int(parsed["winner"])
        except (TypeError, ValueError):
            pass
    for name_key in ("p1_name", "p2_name"):
        if isinstance(parsed.get(name_key), str) and parsed[name_key]:
            meta[name_key] = parsed[name_key]

    return meta


def _analyze_rl_trace(trace_obj: Any) -> tuple[Any, float, int | None]:
    """Sanitize RL turn trace and identify max value swing + worst blunder turn."""
    sanitized = sanitize_json_value(trace_obj)
    if isinstance(sanitized, list):
        turns = sanitized
    elif isinstance(sanitized, dict):
        turns = sanitized.get("turns") or sanitized.get("decisions") or []
    else:
        turns = []

    max_swing = 0.0
    max_swing_turn: int | None = None
    worst_drop = 0.0
    worst_drop_turn: int | None = None
    prev_val: float | None = None

    for idx, entry in enumerate(turns):
        if not isinstance(entry, dict):
            continue
        turn_num = entry.get("turn")
        if turn_num is None:
            turn_num = entry.get("decision_idx", idx)
        try:
            t_int = int(turn_num)
        except (TypeError, ValueError):
            t_int = idx

        vb = _finite_float(entry.get("value_before"))
        va = _finite_float(entry.get("value_after"))
        vd = _finite_float(entry.get("value_delta"))
        val = _finite_float(entry.get("value"))

        signed_delta: float | None = None
        if vd is not None:
            signed_delta = vd
        elif vb is not None and va is not None:
            signed_delta = va - vb
        elif val is not None and prev_val is not None:
            signed_delta = val - prev_val

        if val is not None:
            prev_val = val
        elif va is not None:
            prev_val = va

        if signed_delta is None:
            continue
        swing = abs(signed_delta)
        if swing >= max_swing:
            max_swing = swing
            max_swing_turn = t_int
        if signed_delta < 0 and (-signed_delta) >= worst_drop:
            worst_drop = -signed_delta
            worst_drop_turn = t_int

    blunder_turn = worst_drop_turn if worst_drop_turn is not None else max_swing_turn
    return sanitized, max_swing, blunder_turn


@router.post("/replays")
async def upload_replay(
    request: Request,
    file: UploadFile = File(...),
    trace_file: UploadFile | None = File(None),
    project: str = Form("default"),
    run_id: str | None = Form(None),
    match_id: str | None = Form(None),
    title: str | None = Form(None),
    format: str | None = Form(None),
    step: int | None = Form(None),
    p1_name: str = Form(""),
    p2_name: str = Form(""),
    winner: int | None = Form(None),
    tags_json: str = Form("[]"),
    pinned: bool = Form(False),
    rl_trace_json: str | None = Form(None),
    meta_json: str = Form("{}"),
    max_value_swing: float | None = Form(None),
    blunder_turn: int | None = Form(None),
) -> dict[str, Any]:
    raw_bytes = await file.read()
    filename = file.filename or "replay.bin"
    fmt, mime = _detect_format_and_mime(raw_bytes, filename, format, file.content_type)

    extracted = _extract_cardgame_metadata(raw_bytes) if fmt in ("json", "cardgame_replay_json") else {}
    if extracted.get("format") and (not format or format.strip().lower() == "auto"):
        fmt = extracted["format"]

    try:
        tags = json.loads(tags_json or "[]")
        if not isinstance(tags, list):
            tags = []
    except json.JSONDecodeError:
        tags = []

    try:
        meta = json.loads(meta_json or "{}")
        if not isinstance(meta, dict):
            meta = {}
    except json.JSONDecodeError:
        meta = {}

    blob_store = request.app.state.blob_store
    blob_sha, size_bytes = blob_store.save_bytes("replays", raw_bytes)

    raw_trace_bytes: bytes | None = None
    if trace_file is not None:
        raw_trace_bytes = await trace_file.read()
    elif rl_trace_json:
        raw_trace_bytes = rl_trace_json.encode("utf-8")

    has_rl_trace = False
    trace_sha: str | None = None
    computed_swing = 0.0
    computed_blunder: int | None = None

    if raw_trace_bytes:
        try:
            trace_obj = json.loads(raw_trace_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid RL trace JSON: {exc}") from exc
        sanitized_trace, computed_swing, computed_blunder = _analyze_rl_trace(trace_obj)
        trace_payload = json.dumps(sanitized_trace, allow_nan=False, separators=(",", ":")).encode("utf-8")
        gz_trace = gzip.compress(trace_payload)
        trace_sha, _ = blob_store.save_bytes("traces", gz_trace)
        has_rl_trace = True

    eff_swing = computed_swing
    if max_value_swing is not None and math.isfinite(max_value_swing) and max_value_swing > eff_swing:
        eff_swing = float(max_value_swing)
    eff_blunder = blunder_turn if blunder_turn is not None else computed_blunder

    db = request.app.state.db
    bus = request.app.state.event_bus
    replay = db.insert_replay(
        {
            "project": project,
            "run_id": run_id,
            "match_id": match_id,
            "title": title or Path(filename).stem or filename,
            "format": fmt,
            "content_type": mime,
            "blob_sha256": blob_sha,
            "size_bytes": size_bytes,
            "filename": filename,
            "step": step,
            "seed1": extracted.get("seed1"),
            "seed2": extracted.get("seed2"),
            "version": extracted.get("version"),
            "manifest_hash": extracted.get("manifest_hash", ""),
            "deck1": extracted.get("deck1", []),
            "deck2": extracted.get("deck2", []),
            "config": extracted.get("config", {}),
            "action_count": extracted.get("action_count", 0),
            "winner": winner if winner is not None else extracted.get("winner"),
            "p1_name": p1_name or extracted.get("p1_name", ""),
            "p2_name": p2_name or extracted.get("p2_name", ""),
            "has_rl_trace": has_rl_trace,
            "trace_sha256": trace_sha,
            "max_value_swing": eff_swing,
            "blunder_turn": eff_blunder,
            "tags": tags,
            "pinned": pinned,
            "meta": meta,
        }
    )
    bus.publish("replay_recorded", replay, run_id=replay.get("run_id"))
    return {"replay": replay}


@router.get("/replays")
def list_replays(
    request: Request,
    project: str | None = Query(None),
    run_id: str | None = Query(None),
    match_id: str | None = Query(None),
    format: str | None = Query(None),
    deck: str | None = Query(None),
    tag: str | None = Query(None),
    has_trace: bool | None = Query(None),
    min_swing: float | None = Query(None),
    pinned_only: bool = Query(False),
    q: str | None = Query(None),
    sort_by: str = Query("created_at"),
    order: str = Query("desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    db = request.app.state.db
    rows = db.list_replays(
        project=project,
        run_id=run_id,
        match_id=match_id,
        format=format,
        deck=deck,
        tag=tag,
        has_trace=has_trace,
        min_swing=min_swing,
        pinned_only=pinned_only,
        q=q,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return {"replays": rows}


@router.get("/replays/{replay_id}")
def get_replay(replay_id: str, request: Request) -> dict[str, Any]:
    replay = request.app.state.db.get_replay(replay_id)
    if replay is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return {"replay": replay}


@router.get("/replays/{replay_id}/file")
def download_replay_file(
    replay_id: str,
    request: Request,
    inline: bool = Query(False),
) -> FileResponse:
    replay = request.app.state.db.get_replay(replay_id)
    if replay is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    blob_path = request.app.state.blob_store.get_path("replays", replay["blob_sha256"])
    if blob_path is None or not blob_path.exists():
        raise HTTPException(status_code=404, detail="Replay blob missing on disk")
    filename = replay.get("filename") or f"{replay_id}.{replay.get('format', 'bin')}"
    disposition = "inline" if inline else "attachment"
    return FileResponse(
        path=blob_path,
        media_type=replay.get("content_type") or "application/octet-stream",
        filename=filename,
        content_disposition_type=disposition,
    )


@router.get("/replays/{replay_id}/trace")
def get_replay_trace(replay_id: str, request: Request) -> Response:
    replay = request.app.state.db.get_replay(replay_id)
    if replay is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    trace_sha = replay.get("trace_sha256")
    if not replay.get("has_rl_trace") or not trace_sha:
        raise HTTPException(status_code=404, detail="No RL trace attached to this replay")
    trace_path = request.app.state.blob_store.get_path("traces", trace_sha)
    if trace_path is None or not trace_path.exists():
        raise HTTPException(status_code=404, detail="RL trace blob missing on disk")

    raw_trace = gzip.decompress(trace_path.read_bytes()).decode("utf-8")
    trace_obj = sanitize_json_value(json.loads(raw_trace))
    body = json.dumps(
        {
            "replay_id": replay_id,
            "max_value_swing": replay.get("max_value_swing", 0.0),
            "blunder_turn": replay.get("blunder_turn"),
            "trace": trace_obj,
        },
        allow_nan=False,
    )
    return Response(content=body, media_type="application/json")


@router.patch("/replays/{replay_id}")
def patch_replay(
    replay_id: str, req: ReplayPatchRequest, request: Request
) -> dict[str, Any]:
    updated = request.app.state.db.update_replay(
        replay_id, title=req.title, pinned=req.pinned, tags=req.tags
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return {"replay": updated}


@router.delete("/replays/{replay_id}")
def delete_replay(replay_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    blob_store = request.app.state.blob_store
    deleted = db.delete_replay(replay_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Replay not found")

    refs = db.count_replay_blob_refs(
        blob_sha256=deleted.get("blob_sha256"),
        trace_sha256=deleted.get("trace_sha256"),
    )
    if deleted.get("blob_sha256") and refs["blob_refs"] == 0:
        blob_store.delete("replays", deleted["blob_sha256"])
    if deleted.get("trace_sha256") and refs["trace_refs"] == 0:
        blob_store.delete("traces", deleted["trace_sha256"])
    return {"deleted": replay_id}

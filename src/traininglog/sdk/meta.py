"""SDK helpers for card catalog metadata and per-game meta telemetry."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable

import httpx

from .league import _resolve_url_and_token


def _sanitize_json(val: Any) -> Any:
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, dict):
        return {str(k): _sanitize_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_json(v) for v in val]
    return val


def log_card_catalog(
    cards: list[dict[str, Any]] | dict[str, dict[str, Any]],
    *,
    project: str = "default",
    manifest_hash: str = "",
    patch_label: str = "",
    notes: str = "",
    num_lanes: int = 4,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Upsert optional card catalog definitions versioned by (project, manifest_hash)."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    normalized: list[dict[str, Any]] = []
    if isinstance(cards, dict):
        for card_key, info in cards.items():
            entry = dict(info) if isinstance(info, dict) else {"display_name": str(info)}
            entry.setdefault("card_key", str(card_key))
            normalized.append(entry)
    else:
        for item in cards:
            normalized.append(dict(item))

    payload = _sanitize_json(
        {
            "project": project,
            "manifest_hash": manifest_hash,
            "patch_label": patch_label,
            "notes": notes,
            "num_lanes": int(num_lanes),
            "cards": normalized,
        }
    )
    body = json.dumps(payload, allow_nan=False)
    with httpx.Client(base_url=url, headers=headers, transport=transport, timeout=30.0) as client:
        resp = client.post(
            "/api/meta/catalog",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def log_meta_games(
    games: Iterable[dict[str, Any]],
    *,
    project: str = "default",
    source: str = "league",
    manifest_hash: str = "",
    match_id: str | None = None,
    run_id: str | None = None,
    value_scale: str = "prob",
    gzip_compress: bool = True,
    chunk_size: int = 500,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Upload per-game meta telemetry in chunks (optionally gzip-compressed)."""
    url, headers = _resolve_url_and_token(server_url, api_token)
    eff_chunk = max(1, int(chunk_size))

    total_inserted = 0
    total_duplicates = 0
    total_plays = 0
    last_watermark = 0
    project_id = ""

    batch: list[dict[str, Any]] = []

    def _flush(client: httpx.Client, items: list[dict[str, Any]]) -> None:
        nonlocal total_inserted, total_duplicates, total_plays, last_watermark, project_id
        if not items:
            return
        payload = _sanitize_json(
            {
                "project": project,
                "source": source,
                "manifest_hash": manifest_hash,
                "match_id": match_id,
                "run_id": run_id,
                "value_scale": value_scale,
                "games": items,
            }
        )
        raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if gzip_compress:
            raw = gzip.compress(raw)
            req_headers["Content-Encoding"] = "gzip"
        resp = client.post("/api/meta/games/batch", content=raw, headers=req_headers)
        resp.raise_for_status()
        res = resp.json()
        total_inserted += int(res.get("inserted", 0))
        total_duplicates += int(res.get("duplicates", 0))
        total_plays += int(res.get("total_plays", 0))
        last_watermark = int(res.get("watermark", last_watermark))
        project_id = str(res.get("project_id", project_id))

    with httpx.Client(base_url=url, headers=headers, transport=transport, timeout=60.0) as client:
        for game in games:
            batch.append(dict(game))
            if len(batch) >= eff_chunk:
                _flush(client, batch)
                batch = []
        if batch:
            _flush(client, batch)

    return {
        "project": project,
        "project_id": project_id,
        "inserted": total_inserted,
        "duplicates": total_duplicates,
        "total_plays": total_plays,
        "watermark": last_watermark,
    }


def import_meta_games_ndjson(
    path: str | Path,
    *,
    project: str = "default",
    source: str | None = None,
    manifest_hash: str = "",
    value_scale: str = "prob",
    chunk_size: int = 500,
    gzip_compress: bool = True,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Stream an .ndjson or .ndjson.gz file of game telemetry records to /api/meta/games/batch."""
    src_path = Path(path)
    if not src_path.is_file():
        raise FileNotFoundError(f"NDJSON file not found: {src_path}")

    def _iter_games() -> Iterable[dict[str, Any]]:
        opener = gzip.open if src_path.suffix.lower() == ".gz" else open
        with opener(src_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    if source and "source" not in obj:
                        obj["source"] = source
                    if manifest_hash and "manifest_hash" not in obj:
                        obj["manifest_hash"] = manifest_hash
                    yield obj

    return log_meta_games(
        _iter_games(),
        project=project,
        source=source or "league",
        manifest_hash=manifest_hash,
        value_scale=value_scale,
        gzip_compress=gzip_compress,
        chunk_size=chunk_size,
        server_url=server_url,
        api_token=api_token,
        transport=transport,
    )

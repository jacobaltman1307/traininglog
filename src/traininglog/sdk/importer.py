"""Read-only SQLite importer for existing cardgame/agents/runs/league.db files."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .league import _resolve_url_and_token, log_checkpoint, log_match_series


def _parse_sqlite_ts_ms(ts_val: Any, tie_offset_ms: int = 0) -> int | None:
    if ts_val is None:
        return None
    s = str(ts_val).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000) + tie_offset_ms
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000) + tie_offset_ms
    except ValueError:
        return None


def import_league_db(
    db_path: str | Path,
    *,
    project: str = "cardgame-ppo",
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Import models and chronological matches from a cardgame league.db read-only."""
    resolved_path = Path(db_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"League database not found: {resolved_path}")

    uri = f"file:{resolved_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row

    try:
        model_rows = conn.execute("SELECT * FROM models ORDER BY created_at ASC").fetchall()
        match_rows = conn.execute(
            "SELECT * FROM matches ORDER BY timestamp ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()

    url, headers = _resolve_url_and_token(server_url, api_token)
    models_imported = 0
    matches_imported = 0
    matches_skipped = 0

    for mr in model_rows:
        name = str(mr["name"])
        ckpt_path = str(mr["checkpoint_path"] or "")
        arch = mr["arch"] if "arch" in mr.keys() else None
        tags: list[str] = ["league"]
        if "saved_runs" in ckpt_path:
            tags.append("saved_run")
        log_checkpoint(
            ckpt_path or name,
            project=project,
            name=name,
            arch=arch if arch and arch != "unknown" else None,
            tags=tags,
            server_url=url,
            api_token=api_token,
            transport=transport,
        )
        models_imported += 1

    for row in match_rows:
        m_id = int(row["id"])
        played_ms = _parse_sqlite_ts_ms(row["timestamp"], tie_offset_ms=m_id % 1000)
        res = log_match_series(
            project=project,
            model_a=str(row["model_a"]),
            model_b=str(row["model_b"]),
            total_games=int(row["total_games"]),
            a_wins=int(row["a_wins"]),
            b_wins=int(row["b_wins"]),
            draws=int(row["draws"]),
            a_as_p1_wins=int(row["a_as_p1_wins"]),
            a_as_p1_games=int(row["a_as_p1_games"]),
            a_as_p2_wins=int(row["a_as_p2_wins"]),
            a_as_p2_games=int(row["a_as_p2_games"]),
            b_as_p1_wins=int(row["b_as_p1_wins"]),
            b_as_p1_games=int(row["b_as_p1_games"]),
            b_as_p2_wins=int(row["b_as_p2_wins"]),
            b_as_p2_games=int(row["b_as_p2_games"]),
            avg_rounds=float(row["avg_rounds"]),
            avg_steps=float(row["avg_steps"]),
            source_key=f"league_db:{m_id}",
            played_at=played_ms,
            server_url=url,
            api_token=api_token,
            transport=transport,
        )
        if res.get("duplicate"):
            matches_skipped += 1
        else:
            matches_imported += 1

    return {
        "project": project,
        "models_imported": models_imported,
        "matches_imported": matches_imported,
        "matches_skipped": matches_skipped,
    }

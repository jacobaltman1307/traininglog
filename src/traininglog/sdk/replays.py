"""SDK helpers for Godot replays, RL decision traces, card stats, and distributions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def log_replay(
    path_or_bytes: str | Path | bytes,
    *,
    project: str = "default",
    run_id: str | None = None,
    match_id: str | None = None,
    title: str | None = None,
    format: str | None = None,
    step: int | None = None,
    p1_name: str = "",
    p2_name: str = "",
    winner: int | None = None,
    tags: list[str] | None = None,
    pinned: bool = False,
    rl_trace: list[dict[str, Any]] | dict[str, Any] | str | Path | None = None,
    meta: dict[str, Any] | None = None,
    filename: str | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Upload a Godot ReplayRecord JSON, MP4/WebM recording, or binary replay with optional RL trace."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw_bytes = bytes(path_or_bytes)
        inferred_name = filename or (
            "replay.json" if raw_bytes.lstrip().startswith((b"{", b"[")) else "replay.bin"
        )
    else:
        p = Path(path_or_bytes)
        raw_bytes = p.read_bytes()
        inferred_name = filename or p.name

    mime = "application/octet-stream"
    lower_name = inferred_name.lower()
    if lower_name.endswith(".mp4"):
        mime = "video/mp4"
    elif lower_name.endswith(".webm"):
        mime = "video/webm"
    elif lower_name.endswith(".json"):
        mime = "application/json"

    data: dict[str, Any] = {
        "project": project,
        "p1_name": p1_name,
        "p2_name": p2_name,
        "tags_json": json.dumps(list(tags or []), allow_nan=False),
        "pinned": "true" if pinned else "false",
        "meta_json": json.dumps(_sanitize_json(meta or {}), allow_nan=False),
    }
    if run_id is not None:
        data["run_id"] = run_id
    if match_id is not None:
        data["match_id"] = match_id
    if title is not None:
        data["title"] = title
    if format is not None:
        data["format"] = format
    if step is not None:
        data["step"] = str(int(step))
    if winner is not None:
        data["winner"] = str(int(winner))

    files: dict[str, Any] = {"file": (inferred_name, raw_bytes, mime)}
    if rl_trace is not None:
        if isinstance(rl_trace, (str, Path)) and Path(rl_trace).is_file():
            tp = Path(rl_trace)
            files["trace_file"] = (tp.name, tp.read_bytes(), "application/json")
        elif isinstance(rl_trace, str):
            data["rl_trace_json"] = rl_trace
        else:
            data["rl_trace_json"] = json.dumps(_sanitize_json(rl_trace), allow_nan=False)

    with httpx.Client(base_url=url, headers=headers, transport=transport, timeout=60.0) as client:
        resp = client.post("/api/replays", data=data, files=files)
        resp.raise_for_status()
        return resp.json()["replay"]


def log_card_stats(
    cards: list[dict[str, Any]] | dict[str, Any] | None = None,
    *,
    counts: Any = None,
    card_names: list[str] | None = None,
    drawn: Any = None,
    wins_when_played: Any = None,
    games_when_played: Any = None,
    project: str = "default",
    run_id: str | None = None,
    step: int = 0,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Log per-card play/draw/win statistics at a training step."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    normalized: list[dict[str, Any]] = []
    if cards is not None:
        if isinstance(cards, dict):
            for name, val in cards.items():
                if isinstance(val, dict):
                    normalized.append({"card_name": str(name), **val})
                else:
                    normalized.append({"card_name": str(name), "played": int(val or 0)})
        else:
            normalized = list(cards)
    elif counts is not None:
        counts_seq = list(counts)
        names_seq = (
            list(card_names)
            if card_names is not None
            else [f"card_{i}" for i in range(len(counts_seq))]
        )
        drawn_seq = list(drawn) if isinstance(drawn, (list, tuple)) else None
        wins_seq = list(wins_when_played) if isinstance(wins_when_played, (list, tuple)) else None
        games_seq = list(games_when_played) if isinstance(games_when_played, (list, tuple)) else None
        drawn_map = drawn if isinstance(drawn, dict) else {}
        wins_map = wins_when_played if isinstance(wins_when_played, dict) else {}
        games_map = games_when_played if isinstance(games_when_played, dict) else {}

        for idx, name in enumerate(names_seq):
            if idx >= len(counts_seq):
                break
            c_name = str(name)
            row: dict[str, Any] = {
                "card_name": c_name,
                "played": int(counts_seq[idx]),
            }
            if drawn_seq is not None and idx < len(drawn_seq):
                row["drawn"] = int(drawn_seq[idx])
            elif c_name in drawn_map:
                row["drawn"] = int(drawn_map[c_name])
            if wins_seq is not None and idx < len(wins_seq):
                row["wins_when_played"] = int(wins_seq[idx])
            elif c_name in wins_map:
                row["wins_when_played"] = int(wins_map[c_name])
            if games_seq is not None and idx < len(games_seq):
                row["games_when_played"] = int(games_seq[idx])
            elif c_name in games_map:
                row["games_when_played"] = int(games_map[c_name])
            normalized.append(row)

    payload = _sanitize_json(
        {
            "project": project,
            "run_id": run_id,
            "step": int(step),
            "cards": normalized,
        }
    )
    body = json.dumps(payload, allow_nan=False)
    with httpx.Client(base_url=url, headers=headers, transport=transport, timeout=30.0) as client:
        resp = client.post(
            "/api/cards/stats",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def log_distribution(
    name: str,
    probs: list[float] | dict[str, float] | Any,
    *,
    labels: list[str] | None = None,
    run_id: str,
    step: int = 0,
    entropy: float | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Log discrete action/strategy distribution probabilities over training steps."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    if isinstance(probs, dict):
        eff_labels = [str(k) for k in (labels or probs.keys())]
        eff_probs = [float(v) for v in probs.values()]
    else:
        eff_probs = [float(x) for x in list(probs)]
        eff_labels = (
            [str(k) for k in labels]
            if labels is not None
            else [str(i) for i in range(len(eff_probs))]
        )

    eff_probs = [p if math.isfinite(p) else 0.0 for p in eff_probs]
    eff_ent = float(entropy) if (entropy is not None and math.isfinite(entropy)) else None
    payload = _sanitize_json(
        {
            "run_id": run_id,
            "name": str(name),
            "step": int(step),
            "labels": eff_labels,
            "probs": eff_probs,
            "entropy": eff_ent,
        }
    )
    body = json.dumps(payload, allow_nan=False)
    with httpx.Client(base_url=url, headers=headers, transport=transport, timeout=30.0) as client:
        resp = client.post(
            "/api/distributions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()["distribution"]

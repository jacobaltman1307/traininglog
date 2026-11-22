"""SDK helpers for league matches, ratings, checkpoints, and ReDo plasticity diagnostics."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import httpx


def _resolve_url_and_token(
    server_url: str | None, api_token: str | None
) -> tuple[str, dict[str, str]]:
    url = (
        server_url or os.environ.get("TRAININGLOG_URL", "http://127.0.0.1:8765")
    ).rstrip("/")
    tok = (
        api_token if api_token is not None else os.environ.get("TRAININGLOG_API_TOKEN")
    ) or None
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    return url, headers


def _sha256_small_file(path: Path, max_bytes: int = 100 * 1024 * 1024) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def log_checkpoint(
    path_or_name: str | Path,
    *,
    project: str = "default",
    name: str | None = None,
    step: int = 0,
    run_id: str | None = None,
    parent: str | None = None,
    arch: str | None = None,
    obs_dim: int | None = None,
    manifest_hash: str | None = None,
    tags: list[str] | None = None,
    pinned: bool = False,
    metrics: dict[str, Any] | None = None,
    upload: bool = False,
    compute_hash: bool = False,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Register a checkpoint (path reference by default, or upload=True to store weights)."""
    url, headers = _resolve_url_and_token(server_url, api_token)
    raw_str = str(path_or_name)
    is_builtin = raw_str.startswith("builtin:")
    p = Path(raw_str)
    resolved_name = name or (
        raw_str.split(":", 1)[1] if is_builtin else p.stem or raw_str
    )

    with httpx.Client(
        base_url=url, headers=headers, timeout=60.0, transport=transport
    ) as client:
        if upload:
            if not p.is_file():
                raise FileNotFoundError(f"Checkpoint file not found for upload: {p}")
            with open(p, "rb") as f:
                files = {"file": (p.name, f, "application/octet-stream")}
                data = {
                    "project": project,
                    "name": resolved_name,
                    "step": str(int(step)),
                    "tags_json": json.dumps(list(tags or [])),
                    "pinned": "true" if pinned else "false",
                    "metrics_json": json.dumps(metrics or {}),
                }
                if run_id:
                    data["run_id"] = run_id
                if parent:
                    data["parent_id"] = parent
                if arch:
                    data["arch"] = arch
                if obs_dim is not None:
                    data["obs_dim"] = str(int(obs_dim))
                if manifest_hash:
                    data["manifest_hash"] = manifest_hash
                resp = client.post("/api/checkpoints/upload", files=files, data=data)
                resp.raise_for_status()
                return resp.json()["checkpoint"]

        abs_path = raw_str if is_builtin else str(p.resolve()) if p.exists() else raw_str
        size_bytes = p.stat().st_size if (not is_builtin and p.is_file()) else None
        sha = (
            _sha256_small_file(p, max_bytes=10 * 1024**3 if compute_hash else 100 * 1024 * 1024)
            if (not is_builtin and p.is_file())
            else None
        )
        payload = {
            "project": project,
            "run_id": run_id,
            "name": resolved_name,
            "step": int(step),
            "parent": parent,
            "arch": arch,
            "obs_dim": obs_dim,
            "manifest_hash": manifest_hash,
            "storage_mode": "builtin" if is_builtin else "path_ref",
            "path_or_blob": abs_path,
            "host": socket.gethostname() if not is_builtin else "builtin",
            "sha256": sha,
            "size_bytes": size_bytes,
            "tags": list(tags or []),
            "pinned": pinned,
            "metrics": metrics or {},
        }
        resp = client.post("/api/checkpoints", json=payload)
        resp.raise_for_status()
        return resp.json()["checkpoint"]


def log_match_series(
    *,
    project: str = "default",
    model_a: str,
    model_b: str,
    total_games: int,
    a_wins: int,
    b_wins: int,
    draws: int = 0,
    a_as_p1_wins: int = 0,
    a_as_p1_games: int = 0,
    a_as_p2_wins: int = 0,
    a_as_p2_games: int = 0,
    b_as_p1_wins: int | None = None,
    b_as_p1_games: int | None = None,
    b_as_p2_wins: int | None = None,
    b_as_p2_games: int | None = None,
    deck_a: str | None = None,
    deck_b: str | None = None,
    avg_rounds: float = 0.0,
    avg_steps: float = 0.0,
    duration_sec: float = 0.0,
    step: int | None = None,
    run_id: str | None = None,
    manifest_hash: str | None = None,
    source_key: str | None = None,
    played_at: int | None = None,
    metadata: dict[str, Any] | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Log an aggregated 1v1 match series (e.g., 1,000 games from league.py)."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    resolved_b_p1_games = b_as_p1_games if b_as_p1_games is not None else a_as_p2_games
    resolved_b_p2_games = b_as_p2_games if b_as_p2_games is not None else a_as_p1_games
    if b_as_p1_wins is None:
        resolved_b_p1_wins = min(b_wins, max(0, resolved_b_p1_games - a_as_p2_wins))
    else:
        resolved_b_p1_wins = b_as_p1_wins
    if b_as_p2_wins is None:
        resolved_b_p2_wins = max(0, b_wins - resolved_b_p1_wins)
    else:
        resolved_b_p2_wins = b_as_p2_wins

    payload = {
        "project": project,
        "run_id": run_id,
        "source_key": source_key,
        "step": step,
        "mode": "1v1_series" if total_games > 1 else "1v1",
        "total_games": int(total_games),
        "draws": int(draws),
        "p1_side_wins": int(a_as_p1_wins + resolved_b_p1_wins),
        "p2_side_wins": int(a_as_p2_wins + resolved_b_p2_wins),
        "avg_rounds": float(avg_rounds),
        "avg_steps": float(avg_steps),
        "duration_sec": float(duration_sec),
        "manifest_hash": manifest_hash,
        "metadata": metadata or {},
        "played_at": played_at,
        "participants": [
            {
                "agent": model_a,
                "seat": 0,
                "deck_or_faction": deck_a,
                "rank": 1 if a_wins > b_wins else 2 if a_wins < b_wins else 1,
                "score": float(a_wins + 0.5 * draws),
                "wins": int(a_wins),
                "losses": int(b_wins),
                "draws": int(draws),
                "games": int(total_games),
                "as_p1_wins": int(a_as_p1_wins),
                "as_p1_games": int(a_as_p1_games),
                "as_p2_wins": int(a_as_p2_wins),
                "as_p2_games": int(a_as_p2_games),
            },
            {
                "agent": model_b,
                "seat": 1,
                "deck_or_faction": deck_b,
                "rank": 2 if a_wins > b_wins else 1 if a_wins < b_wins else 1,
                "score": float(b_wins + 0.5 * draws),
                "wins": int(b_wins),
                "losses": int(a_wins),
                "draws": int(draws),
                "games": int(total_games),
                "as_p1_wins": int(resolved_b_p1_wins),
                "as_p1_games": int(resolved_b_p1_games),
                "as_p2_wins": int(resolved_b_p2_wins),
                "as_p2_games": int(resolved_b_p2_games),
            },
        ],
    }

    with httpx.Client(
        base_url=url, headers=headers, timeout=15.0, transport=transport
    ) as client:
        resp = client.post("/api/matches", json=payload)
        resp.raise_for_status()
        return resp.json()


def log_match(
    *,
    project: str = "default",
    participants: list[dict[str, Any]],
    mode: str | None = None,
    turns: int = 0,
    duration_sec: float = 0.0,
    step: int | None = None,
    run_id: str | None = None,
    manifest_hash: str | None = None,
    source_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    played_at: int | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Log a single 1v1, multiplayer FFA, or team match."""
    url, headers = _resolve_url_and_token(server_url, api_token)
    payload = {
        "project": project,
        "run_id": run_id,
        "source_key": source_key,
        "step": step,
        "mode": mode,
        "total_games": 1,
        "avg_rounds": float(turns),
        "duration_sec": float(duration_sec),
        "manifest_hash": manifest_hash,
        "metadata": metadata or {},
        "played_at": played_at,
        "participants": participants,
    }
    with httpx.Client(
        base_url=url, headers=headers, timeout=15.0, transport=transport
    ) as client:
        resp = client.post("/api/matches", json=payload)
        resp.raise_for_status()
        return resp.json()


def log_rating(
    *,
    project: str = "default",
    agent: str,
    glicko_mu: float | None = None,
    glicko_rd: float | None = None,
    elo: float | None = None,
    os_mu: float | None = None,
    os_sigma: float | None = None,
    step: int | None = None,
    run_id: str | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Directly log a pre-computed rating for an agent or checkpoint."""
    url, headers = _resolve_url_and_token(server_url, api_token)
    payload = {
        "project": project,
        "agent": agent,
        "glicko_mu": glicko_mu,
        "glicko_rd": glicko_rd,
        "elo": elo,
        "os_mu": os_mu,
        "os_sigma": os_sigma,
        "step": step,
        "run_id": run_id,
    }
    with httpx.Client(
        base_url=url, headers=headers, timeout=10.0, transport=transport
    ) as client:
        resp = client.post("/api/ratings/log", json=payload)
        resp.raise_for_status()
        return resp.json()


def log_plasticity(
    *,
    project: str = "default",
    layers: list[Any] | None = None,
    meta: dict[str, Any] | None = None,
    json_path: str | Path | None = None,
    model_name: str | None = None,
    checkpoint: str | None = None,
    run_id: str | None = None,
    step: int | None = None,
    arch: str | None = None,
    hook_mode: str | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Log ReDo dormant neuron & effective rank (srank) diagnostics."""
    url, headers = _resolve_url_and_token(server_url, api_token)

    resolved_meta = dict(meta or {})
    resolved_layers = list(layers or [])

    if json_path is not None:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not resolved_meta:
            resolved_meta = dict(raw.get("meta") or {})
        if not resolved_layers:
            resolved_layers = list(raw.get("layers") or [])

    serialized_layers: list[dict[str, Any]] = []
    for lyr in resolved_layers:
        if dataclasses.is_dataclass(lyr):
            d = dataclasses.asdict(lyr)
        elif isinstance(lyr, dict):
            d = dict(lyr)
        else:
            d = dict(vars(lyr))
        if "rank_ratio_pct" not in d and "rank_ratio" in d:
            d["rank_ratio_pct"] = d["rank_ratio"]
        serialized_layers.append(d)

    payload = {
        "project": project,
        "run_id": run_id,
        "checkpoint": checkpoint or resolved_meta.get("model_name") or model_name,
        "model_name": model_name or resolved_meta.get("model_name") or checkpoint or "model",
        "step": step if step is not None else resolved_meta.get("step"),
        "arch": arch or resolved_meta.get("arch"),
        "hook_mode": hook_mode or resolved_meta.get("hook_mode") or "post_activation",
        "total_obs": resolved_meta.get("total_obs"),
        "meta": resolved_meta,
        "layers": serialized_layers,
    }

    with httpx.Client(
        base_url=url, headers=headers, timeout=15.0, transport=transport
    ) as client:
        resp = client.post("/api/plasticity", json=payload)
        resp.raise_for_status()
        return resp.json()

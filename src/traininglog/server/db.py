"""SQLite WAL database layer for TrainingLog."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .ratings import (
    Glicko2Rating,
    OpenSkillRating,
    rate_elo_match,
    rate_glicko2_match,
    rate_openskill_match,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def sanitize_json_value(val: Any) -> Any:
    """Recursively convert non-finite floats into None so JSON stays strict."""
    if isinstance(val, float):
        import math

        return val if math.isfinite(val) else None
    if isinstance(val, dict):
        return {str(k): sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_json_value(v) for v in val]
    return val


class Database:
    """Thread-safe SQLite WAL database wrapper."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA busy_timeout=10000;")
        self.init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def init_schema(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        created_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        group_name TEXT,
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        notes TEXT NOT NULL DEFAULT '',
                        state TEXT NOT NULL DEFAULT 'running',
                        arch TEXT,
                        obs_dim INTEGER,
                        manifest_hash TEXT,
                        config_json TEXT NOT NULL DEFAULT '{}',
                        summary_json TEXT NOT NULL DEFAULT '{}',
                        git_json TEXT NOT NULL DEFAULT '{}',
                        system_info_json TEXT NOT NULL DEFAULT '{}',
                        cli_args TEXT NOT NULL DEFAULT '',
                        host TEXT NOT NULL DEFAULT '',
                        parent_checkpoint_id TEXT,
                        total_timesteps INTEGER,
                        current_step INTEGER NOT NULL DEFAULT 0,
                        current_sps REAL NOT NULL DEFAULT 0.0,
                        last_gpu_json TEXT NOT NULL DEFAULT '{}',
                        started_at INTEGER NOT NULL,
                        last_heartbeat_at INTEGER NOT NULL,
                        finished_at INTEGER
                    );

                    CREATE TABLE IF NOT EXISTS ingested_batches (
                        run_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        received_at INTEGER NOT NULL,
                        PRIMARY KEY (run_id, seq)
                    );

                    CREATE TABLE IF NOT EXISTS histograms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        wall_ms INTEGER NOT NULL,
                        bins_json TEXT NOT NULL,
                        counts_json TEXT NOT NULL,
                        min_val REAL,
                        max_val REAL,
                        mean_val REAL,
                        std_val REAL,
                        total_count INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS console_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        seq INTEGER NOT NULL,
                        stream TEXT NOT NULL,
                        ts_ms INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        UNIQUE(run_id, seq)
                    );

                    CREATE TABLE IF NOT EXISTS anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        project_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        metric TEXT,
                        step INTEGER,
                        value REAL,
                        message TEXT NOT NULL,
                        resolved INTEGER NOT NULL DEFAULT 0,
                        created_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS alert_rules (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        name TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        target_json TEXT NOT NULL,
                        events_json TEXT NOT NULL,
                        thresholds_json TEXT NOT NULL DEFAULT '{}',
                        cooldown_sec INTEGER NOT NULL DEFAULT 300,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        created_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS alert_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rule_id TEXT NOT NULL,
                        run_id TEXT,
                        anomaly_id INTEGER,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        response_code INTEGER,
                        error TEXT,
                        payload_json TEXT,
                        sent_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS run_commands (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        cmd_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at INTEGER NOT NULL,
                        acked_at INTEGER,
                        completed_at INTEGER,
                        result_message TEXT
                    );

                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        name TEXT NOT NULL,
                        step INTEGER NOT NULL DEFAULT 0,
                        parent_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
                        arch TEXT,
                        obs_dim INTEGER,
                        manifest_hash TEXT,
                        is_bot INTEGER NOT NULL DEFAULT 0,
                        storage_mode TEXT NOT NULL DEFAULT 'path_ref',
                        path_or_blob TEXT NOT NULL DEFAULT '',
                        host TEXT NOT NULL DEFAULT '',
                        sha256 TEXT,
                        size_bytes INTEGER,
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        pinned INTEGER NOT NULL DEFAULT 0,
                        metrics_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL,
                        UNIQUE(project_id, name)
                    );

                    CREATE TABLE IF NOT EXISTS matches (
                        id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        source_key TEXT,
                        step INTEGER,
                        mode TEXT NOT NULL DEFAULT '1v1_series',
                        total_games INTEGER NOT NULL DEFAULT 1,
                        draws INTEGER NOT NULL DEFAULT 0,
                        p1_side_wins INTEGER NOT NULL DEFAULT 0,
                        p2_side_wins INTEGER NOT NULL DEFAULT 0,
                        avg_rounds REAL NOT NULL DEFAULT 0.0,
                        avg_steps REAL NOT NULL DEFAULT 0.0,
                        duration_sec REAL NOT NULL DEFAULT 0.0,
                        manifest_hash TEXT,
                        replay_id TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        played_at INTEGER NOT NULL,
                        UNIQUE(project_id, source_key)
                    );

                    CREATE TABLE IF NOT EXISTS match_participants (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        match_id TEXT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                        agent_id TEXT NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        seat INTEGER,
                        team TEXT,
                        deck_or_faction TEXT,
                        rank INTEGER NOT NULL DEFAULT 1,
                        score REAL NOT NULL DEFAULT 0.0,
                        wins INTEGER NOT NULL DEFAULT 0,
                        losses INTEGER NOT NULL DEFAULT 0,
                        draws INTEGER NOT NULL DEFAULT 0,
                        games INTEGER NOT NULL DEFAULT 1,
                        as_p1_wins INTEGER NOT NULL DEFAULT 0,
                        as_p1_games INTEGER NOT NULL DEFAULT 0,
                        as_p2_wins INTEGER NOT NULL DEFAULT 0,
                        as_p2_games INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS agent_ratings (
                        agent_id TEXT PRIMARY KEY REFERENCES checkpoints(id) ON DELETE CASCADE,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        glicko_mu REAL NOT NULL DEFAULT 1500.0,
                        glicko_rd REAL NOT NULL DEFAULT 350.0,
                        glicko_vol REAL NOT NULL DEFAULT 0.06,
                        elo REAL NOT NULL DEFAULT 1500.0,
                        os_mu REAL NOT NULL DEFAULT 25.0,
                        os_sigma REAL NOT NULL DEFAULT 8.333333333333334,
                        os_ordinal REAL NOT NULL DEFAULT 0.0,
                        matches_played INTEGER NOT NULL DEFAULT 0,
                        games_played INTEGER NOT NULL DEFAULT 0,
                        wins INTEGER NOT NULL DEFAULT 0,
                        losses INTEGER NOT NULL DEFAULT 0,
                        draws INTEGER NOT NULL DEFAULT 0,
                        p1_wins INTEGER NOT NULL DEFAULT 0,
                        p1_games INTEGER NOT NULL DEFAULT 0,
                        p2_wins INTEGER NOT NULL DEFAULT 0,
                        p2_games INTEGER NOT NULL DEFAULT 0,
                        total_rounds REAL NOT NULL DEFAULT 0.0,
                        total_steps REAL NOT NULL DEFAULT 0.0,
                        source TEXT NOT NULL DEFAULT 'computed',
                        updated_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS rating_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        agent_id TEXT NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        match_id TEXT REFERENCES matches(id) ON DELETE CASCADE,
                        step INTEGER,
                        glicko_mu REAL NOT NULL,
                        glicko_rd REAL NOT NULL,
                        elo REAL NOT NULL,
                        os_mu REAL NOT NULL,
                        os_sigma REAL NOT NULL,
                        os_ordinal REAL NOT NULL,
                        source TEXT NOT NULL DEFAULT 'computed',
                        recorded_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS plasticity_reports (
                        id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
                        model_name TEXT NOT NULL,
                        step INTEGER,
                        arch TEXT,
                        hook_mode TEXT NOT NULL DEFAULT 'post_activation',
                        total_obs INTEGER NOT NULL DEFAULT 0,
                        total_neurons INTEGER NOT NULL DEFAULT 0,
                        hard_dead_count INTEGER NOT NULL DEFAULT 0,
                        hard_dead_pct REAL NOT NULL DEFAULT 0.0,
                        dormant_count INTEGER NOT NULL DEFAULT 0,
                        dormant_pct REAL NOT NULL DEFAULT 0.0,
                        weak_count INTEGER NOT NULL DEFAULT 0,
                        weak_pct REAL NOT NULL DEFAULT 0.0,
                        mean_rank_ratio_pct REAL NOT NULL DEFAULT 0.0,
                        layers_json TEXT NOT NULL DEFAULT '[]',
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_runs_project_started
                        ON runs(project_id, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_runs_state
                        ON runs(state);
                    CREATE INDEX IF NOT EXISTS idx_hist_run_name_step
                        ON histograms(run_id, name, step);
                    CREATE INDEX IF NOT EXISTS idx_console_run_id
                        ON console_logs(run_id, id);
                    CREATE INDEX IF NOT EXISTS idx_anomalies_run_created
                        ON anomalies(run_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_commands_run_status
                        ON run_commands(run_id, status, created_at);
                    CREATE INDEX IF NOT EXISTS idx_ckpt_project_run
                        ON checkpoints(project_id, run_id, step);
                    CREATE INDEX IF NOT EXISTS idx_matches_project_played
                        ON matches(project_id, played_at ASC);
                    CREATE INDEX IF NOT EXISTS idx_mp_match
                        ON match_participants(match_id);
                    CREATE INDEX IF NOT EXISTS idx_mp_agent
                        ON match_participants(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_rhist_agent_time
                        ON rating_history(agent_id, recorded_at ASC, id ASC);
                    CREATE INDEX IF NOT EXISTS idx_plasticity_project_ckpt
                        ON plasticity_reports(project_id, checkpoint_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_plasticity_run_step
                        ON plasticity_reports(run_id, step);

                    CREATE TABLE IF NOT EXISTS replays (
                        id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        match_id TEXT REFERENCES matches(id) ON DELETE SET NULL,
                        title TEXT NOT NULL,
                        format TEXT NOT NULL,
                        content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                        blob_sha256 TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        filename TEXT NOT NULL DEFAULT '',
                        step INTEGER,
                        seed1 INTEGER,
                        seed2 INTEGER,
                        version INTEGER,
                        manifest_hash TEXT NOT NULL DEFAULT '',
                        deck1_json TEXT NOT NULL DEFAULT '[]',
                        deck2_json TEXT NOT NULL DEFAULT '[]',
                        config_json TEXT NOT NULL DEFAULT '{}',
                        action_count INTEGER NOT NULL DEFAULT 0,
                        winner INTEGER,
                        p1_name TEXT NOT NULL DEFAULT '',
                        p2_name TEXT NOT NULL DEFAULT '',
                        has_rl_trace INTEGER NOT NULL DEFAULT 0,
                        trace_sha256 TEXT,
                        max_value_swing REAL NOT NULL DEFAULT 0.0,
                        blunder_turn INTEGER,
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        pinned INTEGER NOT NULL DEFAULT 0,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS card_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                        step INTEGER NOT NULL,
                        card_name TEXT NOT NULL,
                        played INTEGER NOT NULL DEFAULT 0,
                        drawn INTEGER NOT NULL DEFAULT 0,
                        wins_when_played INTEGER NOT NULL DEFAULT 0,
                        games_when_played INTEGER NOT NULL DEFAULT 0,
                        play_rate REAL,
                        win_rate REAL,
                        extra_json TEXT NOT NULL DEFAULT '{}',
                        recorded_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS distributions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        step INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        labels_json TEXT NOT NULL,
                        probs_json TEXT NOT NULL,
                        entropy REAL,
                        recorded_at INTEGER NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_replays_project_created
                        ON replays(project_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_replays_run_step
                        ON replays(run_id, step);
                    CREATE INDEX IF NOT EXISTS idx_replays_swing
                        ON replays(project_id, max_value_swing DESC);
                    CREATE INDEX IF NOT EXISTS idx_card_stats_proj_run_step
                        ON card_stats(project_id, run_id, step);
                    CREATE INDEX IF NOT EXISTS idx_card_stats_card
                        ON card_stats(project_id, card_name, step);
                    CREATE INDEX IF NOT EXISTS idx_dist_run_name_step
                        ON distributions(run_id, name, step);
                    """
                )

    # ------------------------------------------------------------------
    # Projects & Runs
    # ------------------------------------------------------------------

    def ensure_project(self, name: str, description: str = "") -> dict[str, Any]:
        ts = now_ms()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                return dict(row)
            proj_id = f"proj_{uuid.uuid4().hex[:12]}"
            with self._conn:
                self._conn.execute(
                    "INSERT INTO projects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                    (proj_id, name, description, ts),
                )
            row = self._conn.execute(
                "SELECT * FROM projects WHERE id = ?", (proj_id,)
            ).fetchone()
            assert row is not None
            return dict(row)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.*,
                       COUNT(r.id) AS total_runs,
                       SUM(CASE WHEN r.state = 'running' THEN 1 ELSE 0 END) AS active_runs,
                       SUM(CASE WHEN r.state = 'stalled' THEN 1 ELSE 0 END) AS stalled_runs
                FROM projects p
                LEFT JOIN runs r ON r.project_id = p.id
                GROUP BY p.id
                ORDER BY p.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def create_or_resume_run(self, data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create a run or resume if run_id already exists. Returns (run_dict, created)."""
        proj = self.ensure_project(data["project"])
        run_id = data.get("run_id") or f"run_{uuid.uuid4().hex[:12]}"
        ts = int(data.get("started_at") or now_ms())

        config_json = _json_dumps(sanitize_json_value(data.get("config") or {}))
        tags_json = _json_dumps(list(data.get("tags") or []))
        git_json = _json_dumps(sanitize_json_value(data.get("git") or {}))
        sys_json = _json_dumps(sanitize_json_value(data.get("system_info") or {}))

        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE runs
                        SET state = 'running',
                            last_heartbeat_at = ?,
                            finished_at = NULL
                        WHERE id = ?
                        """,
                        (now_ms(), run_id),
                    )
                updated = self.get_run(run_id)
                assert updated is not None
                return updated, False

            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO runs (
                        id, project_id, name, group_name, tags_json, notes, state,
                        arch, obs_dim, manifest_hash, config_json, summary_json,
                        git_json, system_info_json, cli_args, host,
                        parent_checkpoint_id, total_timesteps, current_step,
                        current_sps, last_gpu_json, started_at, last_heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, '{}', ?, ?, ?, ?, ?, ?, 0, 0.0, '{}', ?, ?)
                    """,
                    (
                        run_id,
                        proj["id"],
                        data.get("name") or run_id,
                        data.get("group_name"),
                        tags_json,
                        data.get("notes") or "",
                        data.get("arch"),
                        data.get("obs_dim"),
                        data.get("manifest_hash"),
                        config_json,
                        git_json,
                        sys_json,
                        data.get("cli_args") or "",
                        data.get("host") or "",
                        data.get("parent_checkpoint_id"),
                        data.get("total_timesteps"),
                        ts,
                        ts,
                    ),
                )
            created_run = self.get_run(run_id)
            assert created_run is not None
            return created_run, True

    def _hydrate_run(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
        d["config"] = json.loads(d.pop("config_json", "{}") or "{}")
        d["summary"] = json.loads(d.pop("summary_json", "{}") or "{}")
        d["git"] = json.loads(d.pop("git_json", "{}") or "{}")
        d["system_info"] = json.loads(d.pop("system_info_json", "{}") or "{}")
        d["last_gpu"] = json.loads(d.pop("last_gpu_json", "{}") or "{}")
        return d

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT r.*,
                       p.name AS project_name,
                       (SELECT COUNT(*) FROM anomalies a WHERE a.run_id = r.id AND a.resolved = 0) AS open_anomaly_count
                FROM runs r
                JOIN projects p ON p.id = r.project_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return self._hydrate_run(row)

    def list_runs(
        self,
        project: str | None = None,
        state: str | None = None,
        group_name: str | None = None,
        arch: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if state:
            states = [s.strip() for s in state.split(",") if s.strip()]
            if states:
                placeholders = ",".join("?" for _ in states)
                clauses.append(f"r.state IN ({placeholders})")
                params.extend(states)
        if group_name:
            clauses.append("r.group_name = ?")
            params.append(group_name)
        if arch:
            clauses.append("r.arch = ?")
            params.append(arch)
        if search:
            clauses.append("(r.name LIKE ? OR r.id LIKE ? OR r.notes LIKE ?)")
            pat = f"%{search}%"
            params.extend([pat, pat, pat])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT r.*,
                   p.name AS project_name,
                   (SELECT COUNT(*) FROM anomalies a WHERE a.run_id = r.id AND a.resolved = 0) AS open_anomaly_count
            FROM runs r
            JOIN projects p ON p.id = r.project_id
            {where}
            ORDER BY
                CASE WHEN r.state IN ('running', 'stalled') THEN 0 ELSE 1 END,
                r.started_at DESC
            LIMIT ?
        """
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            result = [self._hydrate_run(r) for r in rows]
        if tag:
            result = [r for r in result if tag in r["tags"]]
        return result

    def update_run_metadata(self, run_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            existing = self.get_run(run_id)
            if existing is None:
                return None
            fields: list[str] = []
            params: list[Any] = []
            if "name" in updates and updates["name"] is not None:
                fields.append("name = ?")
                params.append(updates["name"])
            if "group_name" in updates:
                fields.append("group_name = ?")
                params.append(updates["group_name"])
            if "notes" in updates and updates["notes"] is not None:
                fields.append("notes = ?")
                params.append(updates["notes"])
            if "tags" in updates and updates["tags"] is not None:
                fields.append("tags_json = ?")
                params.append(_json_dumps(list(updates["tags"])))
            if "state" in updates and updates["state"] is not None:
                fields.append("state = ?")
                params.append(updates["state"])
            if fields:
                params.append(run_id)
                with self._conn:
                    self._conn.execute(
                        f"UPDATE runs SET {', '.join(fields)} WHERE id = ?",
                        params,
                    )
            return self.get_run(run_id)

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            with self._conn:
                cur = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                self._conn.execute("DELETE FROM ingested_batches WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0

    def check_and_mark_batch(self, run_id: str, seq: int) -> bool:
        """Record batch seq idempotently. Returns True if new, False if duplicate."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO ingested_batches (run_id, seq, received_at) VALUES (?, ?, ?)",
                        (run_id, seq, now_ms()),
                    )
                return True
            except sqlite3.IntegrityError:
                return False

    def filter_and_mark_event_seqs(self, run_id: str, seqs: list[int]) -> set[int]:
        """Record per-event seqs idempotently and return the subset that were newly seen."""
        if not seqs:
            return set()
        min_s = min(seqs)
        max_s = max(seqs)
        ts = now_ms()
        with self._lock:
            existing_rows = self._conn.execute(
                "SELECT seq FROM ingested_batches WHERE run_id = ? AND seq BETWEEN ? AND ?",
                (run_id, min_s, max_s),
            ).fetchall()
            existing = {int(r["seq"]) for r in existing_rows}
            new_seqs = [s for s in seqs if s not in existing]
            if new_seqs:
                with self._conn:
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO ingested_batches (run_id, seq, received_at) VALUES (?, ?, ?)",
                        [(run_id, int(s), ts) for s in new_seqs],
                    )
            return set(new_seqs)

    def apply_run_ingest_state(
        self,
        run_id: str,
        *,
        max_step: int | None = None,
        current_sps: float | None = None,
        summary_updates: dict[str, Any] | None = None,
        config_updates: dict[str, Any] | None = None,
        gpu_updates: dict[str, Any] | None = None,
        heartbeat_ms: int | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Update run summary/config/step/heartbeat atomically. Returns (run, revived_from_stall)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None, False

            revived = False
            cur_state = row["state"]
            new_state = cur_state
            if cur_state == "stalled" and heartbeat_ms is not None:
                new_state = "running"
                revived = True

            cur_step = int(row["current_step"] or 0)
            if max_step is not None and max_step > cur_step:
                cur_step = int(max_step)

            sps = float(row["current_sps"] or 0.0)
            if current_sps is not None:
                sps = float(current_sps)

            summary = json.loads(row["summary_json"] or "{}")
            if summary_updates:
                summary.update(sanitize_json_value(summary_updates))

            config = json.loads(row["config_json"] or "{}")
            if config_updates:
                config.update(sanitize_json_value(config_updates))

            gpu = json.loads(row["last_gpu_json"] or "{}")
            if gpu_updates:
                gpu.update(sanitize_json_value(gpu_updates))

            hb = max(int(row["last_heartbeat_at"] or 0), int(heartbeat_ms or now_ms()))

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE runs
                    SET state = ?,
                        current_step = ?,
                        current_sps = ?,
                        summary_json = ?,
                        config_json = ?,
                        last_gpu_json = ?,
                        last_heartbeat_at = ?
                    WHERE id = ?
                    """,
                    (
                        new_state,
                        cur_step,
                        sps,
                        _json_dumps(summary),
                        _json_dumps(config),
                        _json_dumps(gpu),
                        hb,
                        run_id,
                    ),
                )
                if revived:
                    self._conn.execute(
                        "UPDATE anomalies SET resolved = 1 WHERE run_id = ? AND kind = 'stalled' AND resolved = 0",
                        (run_id,),
                    )
            return self.get_run(run_id), revived

    def finish_run(
        self,
        run_id: str,
        state: str = "finished",
        summary_updates: dict[str, Any] | None = None,
        finished_at: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            summary = json.loads(row["summary_json"] or "{}")
            if summary_updates:
                summary.update(sanitize_json_value(summary_updates))
            fin_ts = int(finished_at or now_ms())
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE runs
                    SET state = ?,
                        summary_json = ?,
                        finished_at = ?,
                        last_heartbeat_at = ?
                    WHERE id = ?
                    """,
                    (state, _json_dumps(summary), fin_ts, max(int(row["last_heartbeat_at"] or 0), fin_ts), run_id),
                )
            return self.get_run(run_id)

    def find_stale_running_runs(self, cutoff_ms: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT r.*, p.name AS project_name, 0 AS open_anomaly_count
                FROM runs r
                JOIN projects p ON p.id = r.project_id
                WHERE r.state = 'running' AND r.last_heartbeat_at < ?
                """,
                (cutoff_ms,),
            ).fetchall()
            return [self._hydrate_run(r) for r in rows]

    def mark_run_stalled(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE runs SET state = 'stalled' WHERE id = ? AND state = 'running'",
                    (run_id,),
                )
            return self.get_run(run_id)

    # ------------------------------------------------------------------
    # Histograms & Console Logs
    # ------------------------------------------------------------------

    def insert_histograms(self, run_id: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO histograms (
                        run_id, name, step, wall_ms, bins_json, counts_json,
                        min_val, max_val, mean_val, std_val, total_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            it["name"],
                            int(it["step"]),
                            int(it.get("wall_ms") or now_ms()),
                            _json_dumps(sanitize_json_value(it["bins"])),
                            _json_dumps(it["counts"]),
                            sanitize_json_value(it.get("min_val")),
                            sanitize_json_value(it.get("max_val")),
                            sanitize_json_value(it.get("mean_val")),
                            sanitize_json_value(it.get("std_val")),
                            int(it.get("total_count", sum(it["counts"]))),
                        )
                        for it in items
                    ],
                )

    def list_histogram_names(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT name, COUNT(*) AS snapshots, MIN(step) AS min_step, MAX(step) AS max_step
                FROM histograms
                WHERE run_id = ?
                GROUP BY name
                ORDER BY name
                """,
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_histograms(self, run_id: str, name: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM histograms
                WHERE run_id = ? AND name = ?
                ORDER BY step DESC
                LIMIT ?
                """,
                (run_id, name, limit),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in reversed(rows):
                d = dict(r)
                d["bins"] = json.loads(d.pop("bins_json"))
                d["counts"] = json.loads(d.pop("counts_json"))
                out.append(d)
            return out

    def insert_console_logs(self, run_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        inserted: list[dict[str, Any]] = []
        with self._lock:
            with self._conn:
                for it in items:
                    cur = self._conn.execute(
                        """
                        INSERT OR IGNORE INTO console_logs (run_id, seq, stream, ts_ms, text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            int(it["seq"]),
                            it.get("stream", "stdout"),
                            int(it.get("ts_ms") or now_ms()),
                            str(it.get("text", "")),
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted.append(
                            {
                                "id": cur.lastrowid,
                                "run_id": run_id,
                                "seq": int(it["seq"]),
                                "stream": it.get("stream", "stdout"),
                                "ts_ms": int(it.get("ts_ms") or now_ms()),
                                "text": str(it.get("text", "")),
                            }
                        )
        return inserted

    def get_console_logs(
        self,
        run_id: str,
        after_id: int = 0,
        limit: int = 500,
        stream: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id = ?", "id > ?"]
        params: list[Any] = [run_id, after_id]
        if stream:
            clauses.append("stream = ?")
            params.append(stream)
        where = " AND ".join(clauses)
        with self._lock:
            if after_id == 0:
                rows = self._conn.execute(
                    f"""
                    SELECT * FROM (
                        SELECT * FROM console_logs WHERE {where} ORDER BY id DESC LIMIT ?
                    ) sub ORDER BY id ASC
                    """,
                    (*params, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT * FROM console_logs WHERE {where} ORDER BY id ASC LIMIT ?",
                    (*params, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Anomalies & Alerts
    # ------------------------------------------------------------------

    def insert_anomaly(
        self,
        *,
        run_id: str,
        project_id: str,
        kind: str,
        severity: str,
        message: str,
        metric: str | None = None,
        step: int | None = None,
        value: float | None = None,
    ) -> dict[str, Any]:
        ts = now_ms()
        clean_val = sanitize_json_value(value)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO anomalies (
                        run_id, project_id, kind, severity, metric, step, value, message, resolved, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (run_id, project_id, kind, severity, metric, step, clean_val, message, ts),
                )
                row = self._conn.execute(
                    "SELECT * FROM anomalies WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
                assert row is not None
                return dict(row)

    def list_anomalies(
        self,
        *,
        run_id: str | None = None,
        project_id: str | None = None,
        resolved: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("a.run_id = ?")
            params.append(run_id)
        if project_id:
            clauses.append("(a.project_id = ? OR p.name = ?)")
            params.extend([project_id, project_id])
        if resolved is not None:
            clauses.append("a.resolved = ?")
            params.append(int(resolved))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT a.*, r.name AS run_name, p.name AS project_name
                FROM anomalies a
                JOIN runs r ON r.id = a.run_id
                JOIN projects p ON p.id = a.project_id
                {where}
                ORDER BY a.created_at DESC, a.id DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_anomaly(self, anomaly_id: int) -> dict[str, Any] | None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE anomalies SET resolved = 1 WHERE id = ?", (anomaly_id,)
                )
            row = self._conn.execute(
                "SELECT * FROM anomalies WHERE id = ?", (anomaly_id,)
            ).fetchone()
            return dict(row) if row else None

    def _hydrate_rule(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["target"] = json.loads(d.pop("target_json"))
        d["events"] = json.loads(d.pop("events_json"))
        d["thresholds"] = json.loads(d.pop("thresholds_json") or "{}")
        d["enabled"] = bool(d["enabled"])
        return d

    def create_alert_rule(self, data: dict[str, Any]) -> dict[str, Any]:
        rule_id = data.get("id") or f"rule_{uuid.uuid4().hex[:10]}"
        ts = now_ms()
        project_id = data.get("project_id")
        if project_id and not project_id.startswith("proj_"):
            proj = self.ensure_project(project_id)
            project_id = proj["id"]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO alert_rules (
                        id, project_id, name, channel, target_json, events_json,
                        thresholds_json, cooldown_sec, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id,
                        project_id,
                        data["name"],
                        data["channel"],
                        _json_dumps(data["target"]),
                        _json_dumps(list(data["events"])),
                        _json_dumps(data.get("thresholds") or {}),
                        int(data.get("cooldown_sec", 300)),
                        1 if data.get("enabled", True) else 0,
                        ts,
                    ),
                )
            row = self._conn.execute(
                "SELECT * FROM alert_rules WHERE id = ?", (rule_id,)
            ).fetchone()
            assert row is not None
            return self._hydrate_rule(row)

    def get_alert_rule(self, rule_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM alert_rules WHERE id = ?", (rule_id,)
            ).fetchone()
            return self._hydrate_rule(row) if row else None

    def list_alert_rules(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM alert_rules"
            if enabled_only:
                sql += " WHERE enabled = 1"
            sql += " ORDER BY created_at DESC"
            rows = self._conn.execute(sql).fetchall()
            return [self._hydrate_rule(r) for r in rows]

    def update_alert_rule(self, rule_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            existing = self.get_alert_rule(rule_id)
            if existing is None:
                return None
            fields: list[str] = []
            params: list[Any] = []
            if "name" in updates and updates["name"] is not None:
                fields.append("name = ?")
                params.append(updates["name"])
            if "channel" in updates and updates["channel"] is not None:
                fields.append("channel = ?")
                params.append(updates["channel"])
            if "target" in updates and updates["target"] is not None:
                fields.append("target_json = ?")
                params.append(_json_dumps(updates["target"]))
            if "events" in updates and updates["events"] is not None:
                fields.append("events_json = ?")
                params.append(_json_dumps(list(updates["events"])))
            if "thresholds" in updates and updates["thresholds"] is not None:
                fields.append("thresholds_json = ?")
                params.append(_json_dumps(updates["thresholds"]))
            if "cooldown_sec" in updates and updates["cooldown_sec"] is not None:
                fields.append("cooldown_sec = ?")
                params.append(int(updates["cooldown_sec"]))
            if "enabled" in updates and updates["enabled"] is not None:
                fields.append("enabled = ?")
                params.append(1 if updates["enabled"] else 0)
            if fields:
                params.append(rule_id)
                with self._conn:
                    self._conn.execute(
                        f"UPDATE alert_rules SET {', '.join(fields)} WHERE id = ?",
                        params,
                    )
            return self.get_alert_rule(rule_id)

    def delete_alert_rule(self, rule_id: str) -> bool:
        with self._lock:
            with self._conn:
                cur = self._conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
            return cur.rowcount > 0

    def last_alert_sent_at(self, rule_id: str, run_id: str | None, event_type: str) -> int | None:
        with self._lock:
            if run_id is None:
                row = self._conn.execute(
                    """
                    SELECT MAX(sent_at) AS ts FROM alert_events
                    WHERE rule_id = ? AND run_id IS NULL AND event_type = ? AND status = 'sent'
                    """,
                    (rule_id, event_type),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """
                    SELECT MAX(sent_at) AS ts FROM alert_events
                    WHERE rule_id = ? AND run_id = ? AND event_type = ? AND status = 'sent'
                    """,
                    (rule_id, run_id, event_type),
                ).fetchone()
            return int(row["ts"]) if row and row["ts"] is not None else None

    def record_alert_event(
        self,
        *,
        rule_id: str,
        run_id: str | None,
        anomaly_id: int | None,
        event_type: str,
        status: str,
        response_code: int | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts = now_ms()
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO alert_events (
                        rule_id, run_id, anomaly_id, event_type, status, response_code, error, payload_json, sent_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id,
                        run_id,
                        anomaly_id,
                        event_type,
                        status,
                        response_code,
                        error,
                        _json_dumps(sanitize_json_value(payload or {})),
                        ts,
                    ),
                )
                row = self._conn.execute(
                    "SELECT * FROM alert_events WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
                assert row is not None
                return dict(row)

    def list_alert_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.*, r.name AS rule_name, r.channel AS channel
                FROM alert_events e
                LEFT JOIN alert_rules r ON r.id = e.rule_id
                ORDER BY e.sent_at DESC, e.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["payload"] = json.loads(d.pop("payload_json") or "{}")
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # Cooperative Run Commands
    # ------------------------------------------------------------------

    def _hydrate_command(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json") or "{}")
        return d

    def create_command(
        self, run_id: str, cmd_type: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cmd_id = f"cmd_{uuid.uuid4().hex[:12]}"
        ts = now_ms()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO run_commands (id, run_id, cmd_type, payload_json, status, created_at)
                    VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (cmd_id, run_id, cmd_type, _json_dumps(payload or {}), ts),
                )
            row = self._conn.execute(
                "SELECT * FROM run_commands WHERE id = ?", (cmd_id,)
            ).fetchone()
            assert row is not None
            return self._hydrate_command(row)

    def list_commands(
        self, run_id: str, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    """
                    SELECT * FROM run_commands
                    WHERE run_id = ? AND status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (run_id, status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM run_commands
                    WHERE run_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (run_id, limit),
                ).fetchall()
            return [self._hydrate_command(r) for r in rows]

    def ack_command(
        self,
        run_id: str,
        cmd_id: str,
        status: str = "acked",
        result_message: str | None = None,
    ) -> dict[str, Any] | None:
        ts = now_ms()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM run_commands WHERE id = ? AND run_id = ?",
                (cmd_id, run_id),
            ).fetchone()
            if row is None:
                return None
            acked_at = row["acked_at"] or ts
            completed_at = ts if status == "completed" else row["completed_at"]
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE run_commands
                    SET status = ?,
                        acked_at = ?,
                        completed_at = ?,
                        result_message = COALESCE(?, result_message)
                    WHERE id = ? AND run_id = ?
                    """,
                    (status, acked_at, completed_at, result_message, cmd_id, run_id),
                )
            updated = self._conn.execute(
                "SELECT * FROM run_commands WHERE id = ?", (cmd_id,)
            ).fetchone()
            assert updated is not None
            return self._hydrate_command(updated)

    # ------------------------------------------------------------------
    # Checkpoints & Lineage
    # ------------------------------------------------------------------

    @staticmethod
    def is_bot_checkpoint(name: str, path_or_blob: str = "", storage_mode: str = "") -> bool:
        return (
            storage_mode == "builtin"
            or path_or_blob.startswith("builtin:")
            or name.startswith(("Random", "Heuristic", "Advanced"))
        )

    @staticmethod
    def is_saved_or_named_checkpoint(row: dict[str, Any]) -> bool:
        if row.get("is_bot") or row.get("pinned"):
            return True
        path_or_blob = str(row.get("path_or_blob") or "")
        tags = row.get("tags") or []
        name = str(row.get("name") or "")
        if "saved_runs" in path_or_blob or "saved_run" in tags or "league" in tags or "best" in tags:
            return True
        if name == "ppo" or name.startswith("cardgame-v0__ppo__") or "cardgame-v0__ppo__" in path_or_blob:
            return False
        return len(tags) > 0 or row.get("storage_mode") == "uploaded"

    def _resolve_project_id(self, project: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM projects WHERE id = ? OR name = ?",
                (project, project),
            ).fetchone()
            return str(row["id"]) if row else None

    def _hydrate_checkpoint(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["is_bot"] = bool(d.get("is_bot"))
        d["pinned"] = bool(d.get("pinned"))
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
        d["metrics"] = json.loads(d.pop("metrics_json", "{}") or "{}")
        return d

    def _ensure_rating_row_locked(self, agent_id: str, project_id: str, ts: int) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO agent_ratings (agent_id, project_id, updated_at)
            VALUES (?, ?, ?)
            """,
            (agent_id, project_id, ts),
        )

    def _resolve_checkpoint_ref_locked(self, project_id: str, ref: str | None) -> str | None:
        if not ref:
            return None
        row = self._conn.execute(
            "SELECT id FROM checkpoints WHERE project_id = ? AND (id = ? OR name = ?)",
            (project_id, ref, ref),
        ).fetchone()
        return str(row["id"]) if row else None

    def upsert_checkpoint(self, data: dict[str, Any]) -> dict[str, Any]:
        proj = self.ensure_project(data.get("project") or "default")
        project_id = proj["id"]
        name = str(data["name"])
        path_or_blob = str(data.get("path_or_blob") or "")
        storage_mode = str(data.get("storage_mode") or ("builtin" if path_or_blob.startswith("builtin:") else "path_ref"))
        is_bot = 1 if (data.get("is_bot") or self.is_bot_checkpoint(name, path_or_blob, storage_mode)) else 0
        tags = list(data.get("tags") or [])
        if is_bot and "bot" not in tags:
            tags.append("bot")
        metrics = sanitize_json_value(data.get("metrics") or {})
        ts = int(data.get("created_at") or now_ms())

        with self._lock:
            run_id = data.get("run_id")
            if run_id:
                r_exists = self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone()
                if not r_exists:
                    run_id = None

            parent_id = self._resolve_checkpoint_ref_locked(
                project_id, data.get("parent_id") or data.get("parent")
            )

            existing = self._conn.execute(
                "SELECT * FROM checkpoints WHERE project_id = ? AND name = ?",
                (project_id, name),
            ).fetchone()
            if existing is None and data.get("id"):
                existing = self._conn.execute(
                    "SELECT * FROM checkpoints WHERE id = ?",
                    (data["id"],),
                ).fetchone()

            if existing is not None:
                ckpt_id = str(existing["id"])
                cur_tags = json.loads(existing["tags_json"] or "[]")
                merged_tags = list(dict.fromkeys(cur_tags + tags))
                cur_metrics = json.loads(existing["metrics_json"] or "{}")
                cur_metrics.update(metrics)
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE checkpoints
                        SET run_id = COALESCE(?, run_id),
                            step = CASE WHEN ? > 0 THEN ? ELSE step END,
                            parent_id = COALESCE(?, parent_id),
                            arch = COALESCE(?, arch),
                            obs_dim = COALESCE(?, obs_dim),
                            manifest_hash = COALESCE(?, manifest_hash),
                            is_bot = CASE WHEN ? = 1 THEN 1 ELSE is_bot END,
                            storage_mode = CASE WHEN ? != 'path_ref' OR storage_mode = '' THEN ? ELSE storage_mode END,
                            path_or_blob = CASE WHEN ? != '' THEN ? ELSE path_or_blob END,
                            host = CASE WHEN ? != '' THEN ? ELSE host END,
                            sha256 = COALESCE(?, sha256),
                            size_bytes = COALESCE(?, size_bytes),
                            tags_json = ?,
                            pinned = CASE WHEN ? = 1 THEN 1 ELSE pinned END,
                            metrics_json = ?
                        WHERE id = ?
                        """,
                        (
                            run_id,
                            int(data.get("step") or 0),
                            int(data.get("step") or 0),
                            parent_id,
                            data.get("arch"),
                            data.get("obs_dim"),
                            data.get("manifest_hash"),
                            is_bot,
                            storage_mode,
                            storage_mode,
                            path_or_blob,
                            path_or_blob,
                            str(data.get("host") or ""),
                            str(data.get("host") or ""),
                            data.get("sha256"),
                            data.get("size_bytes"),
                            _json_dumps(merged_tags),
                            1 if data.get("pinned") else 0,
                            _json_dumps(cur_metrics),
                            ckpt_id,
                        ),
                    )
                    self._ensure_rating_row_locked(ckpt_id, project_id, ts)
                res = self.get_checkpoint(ckpt_id)
                assert res is not None
                return res

            ckpt_id = str(data.get("id") or f"ckpt_{uuid.uuid4().hex[:12]}")
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO checkpoints (
                        id, project_id, run_id, name, step, parent_id, arch, obs_dim,
                        manifest_hash, is_bot, storage_mode, path_or_blob, host, sha256,
                        size_bytes, tags_json, pinned, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ckpt_id,
                        project_id,
                        run_id,
                        name,
                        int(data.get("step") or 0),
                        parent_id,
                        data.get("arch"),
                        data.get("obs_dim"),
                        data.get("manifest_hash"),
                        is_bot,
                        storage_mode,
                        path_or_blob,
                        str(data.get("host") or ""),
                        data.get("sha256"),
                        data.get("size_bytes"),
                        _json_dumps(tags),
                        1 if data.get("pinned") else 0,
                        _json_dumps(metrics),
                        ts,
                    ),
                )
                self._ensure_rating_row_locked(ckpt_id, project_id, ts)
            res = self.get_checkpoint(ckpt_id)
            assert res is not None
            return res

    def get_checkpoint(self, ckpt_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT c.*, p.name AS project_name,
                       ar.glicko_mu, ar.glicko_rd, ar.glicko_vol, ar.elo,
                       ar.os_mu, ar.os_sigma, ar.os_ordinal,
                       ar.matches_played, ar.games_played, ar.wins, ar.losses, ar.draws,
                       ar.p1_wins, ar.p1_games, ar.p2_wins, ar.p2_games,
                       r.name AS run_name,
                       pc.name AS parent_name
                FROM checkpoints c
                JOIN projects p ON p.id = c.project_id
                LEFT JOIN agent_ratings ar ON ar.agent_id = c.id
                LEFT JOIN runs r ON r.id = c.run_id
                LEFT JOIN checkpoints pc ON pc.id = c.parent_id
                WHERE c.id = ?
                """,
                (ckpt_id,),
            ).fetchone()
            if row is None:
                return None
            d = self._hydrate_checkpoint(row)
            latest_plast = self._conn.execute(
                """
                SELECT id, hard_dead_pct, dormant_pct, weak_pct, mean_rank_ratio_pct, created_at
                FROM plasticity_reports
                WHERE checkpoint_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (ckpt_id,),
            ).fetchone()
            d["latest_plasticity"] = dict(latest_plast) if latest_plast else None
            return d

    def get_checkpoint_by_name(self, project: str, name: str) -> dict[str, Any] | None:
        proj_id = self._resolve_project_id(project)
        if not proj_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM checkpoints WHERE project_id = ? AND name = ?",
                (proj_id, name),
            ).fetchone()
            return self.get_checkpoint(str(row["id"])) if row else None

    def list_checkpoints(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        arch: str | None = None,
        manifest_hash: str | None = None,
        tag: str | None = None,
        pinned: bool | None = None,
        include_bots: bool = True,
        saved_only: bool = False,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("c.run_id = ?")
            params.append(run_id)
        if arch:
            clauses.append("c.arch = ?")
            params.append(arch)
        if manifest_hash:
            clauses.append("c.manifest_hash = ?")
            params.append(manifest_hash)
        if pinned is not None:
            clauses.append("c.pinned = ?")
            params.append(1 if pinned else 0)
        if not include_bots:
            clauses.append("c.is_bot = 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.*, p.name AS project_name,
                       ar.glicko_mu, ar.glicko_rd, ar.glicko_vol, ar.elo,
                       ar.os_mu, ar.os_sigma, ar.os_ordinal,
                       ar.matches_played, ar.games_played, ar.wins, ar.losses, ar.draws,
                       ar.p1_wins, ar.p1_games, ar.p2_wins, ar.p2_games,
                       r.name AS run_name,
                       pc.name AS parent_name
                FROM checkpoints c
                JOIN projects p ON p.id = c.project_id
                LEFT JOIN agent_ratings ar ON ar.agent_id = c.id
                LEFT JOIN runs r ON r.id = c.run_id
                LEFT JOIN checkpoints pc ON pc.id = c.parent_id
                {where}
                ORDER BY c.pinned DESC, COALESCE(ar.glicko_mu, 1500.0) DESC, c.created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                d = self._hydrate_checkpoint(row)
                if tag and tag not in d["tags"]:
                    continue
                if saved_only and not self.is_saved_or_named_checkpoint(d):
                    continue
                latest_plast = self._conn.execute(
                    """
                    SELECT id, hard_dead_pct, dormant_pct, weak_pct, mean_rank_ratio_pct, created_at
                    FROM plasticity_reports
                    WHERE checkpoint_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (d["id"],),
                ).fetchone()
                d["latest_plasticity"] = dict(latest_plast) if latest_plast else None
                out.append(d)
            return out

    def update_checkpoint(self, ckpt_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            existing = self.get_checkpoint(ckpt_id)
            if existing is None:
                return None
            fields: list[str] = []
            params: list[Any] = []
            if "tags" in updates and updates["tags"] is not None:
                fields.append("tags_json = ?")
                params.append(_json_dumps(list(dict.fromkeys(updates["tags"]))))
            if "pinned" in updates and updates["pinned"] is not None:
                fields.append("pinned = ?")
                params.append(1 if updates["pinned"] else 0)
            if "metrics" in updates and updates["metrics"] is not None:
                merged = dict(existing["metrics"])
                merged.update(sanitize_json_value(updates["metrics"]))
                fields.append("metrics_json = ?")
                params.append(_json_dumps(merged))
            if "arch" in updates:
                fields.append("arch = ?")
                params.append(updates["arch"])
            if "manifest_hash" in updates:
                fields.append("manifest_hash = ?")
                params.append(updates["manifest_hash"])
            if "parent_id" in updates:
                pid = self._resolve_checkpoint_ref_locked(
                    existing["project_id"], updates["parent_id"]
                )
                fields.append("parent_id = ?")
                params.append(pid)
            if fields:
                params.append(ckpt_id)
                with self._conn:
                    self._conn.execute(
                        f"UPDATE checkpoints SET {', '.join(fields)} WHERE id = ?",
                        params,
                    )
            return self.get_checkpoint(ckpt_id)

    def delete_checkpoint(self, ckpt_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self.get_checkpoint(ckpt_id)
            if existing is None:
                return None
            with self._conn:
                self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (ckpt_id,))
            return existing

    def get_checkpoint_lineage(self, project: str | None = None) -> dict[str, Any]:
        nodes = self.list_checkpoints(project=project, include_bots=False, limit=1000)
        node_ids = {n["id"] for n in nodes}
        edges = [
            {"from": n["parent_id"], "to": n["id"]}
            for n in nodes
            if n.get("parent_id") and n["parent_id"] in node_ids
        ]
        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # League Matches & Ratings (Glicko-2 + Elo + OpenSkill)
    # ------------------------------------------------------------------

    def _apply_match_ratings_locked(
        self,
        *,
        project_id: str,
        match_id: str,
        mode: str,
        total_games: int,
        draws: int,
        avg_rounds: float,
        avg_steps: float,
        step: int | None,
        run_id: str | None,
        played_at: int,
        part_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update agent_ratings & rating_history for an already-inserted match inside lock."""
        ratings_by_id: dict[str, dict[str, Any]] = {}
        for p in part_rows:
            aid = p["agent_id"]
            self._ensure_rating_row_locked(aid, project_id, played_at)
            row = self._conn.execute(
                "SELECT * FROM agent_ratings WHERE agent_id = ?", (aid,)
            ).fetchone()
            assert row is not None
            ratings_by_id[aid] = dict(row)

        updates: list[dict[str, Any]] = []
        distinct_agents = list(dict.fromkeys(p["agent_id"] for p in part_rows))

        is_two_party = (
            len(part_rows) == 2
            and mode in ("1v1", "1v1_series")
            and part_rows[0]["agent_id"] != part_rows[1]["agent_id"]
        )

        if is_two_party:
            pa, pb = part_rows[0], part_rows[1]
            ra_row = ratings_by_id[pa["agent_id"]]
            rb_row = ratings_by_id[pb["agent_id"]]

            score_a = (pa["wins"] + 0.5 * draws) / max(total_games, 1)

            ga_after, gb_after = rate_glicko2_match(
                Glicko2Rating(
                    mu=float(ra_row["glicko_mu"]),
                    rd=float(ra_row["glicko_rd"]),
                    vol=float(ra_row["glicko_vol"]),
                ),
                Glicko2Rating(
                    mu=float(rb_row["glicko_mu"]),
                    rd=float(rb_row["glicko_rd"]),
                    vol=float(rb_row["glicko_vol"]),
                ),
                score_a,
            )
            elo_a_after, elo_b_after = rate_elo_match(
                float(ra_row["elo"]), float(rb_row["elo"]), score_a
            )

            rank_a = 1 if score_a > 0.5 else 2 if score_a < 0.5 else 1
            rank_b = 2 if score_a > 0.5 else 1 if score_a < 0.5 else 1
            os_teams = rate_openskill_match(
                [
                    [OpenSkillRating(mu=float(ra_row["os_mu"]), sigma=float(ra_row["os_sigma"]))],
                    [OpenSkillRating(mu=float(rb_row["os_mu"]), sigma=float(rb_row["os_sigma"]))],
                ],
                [rank_a, rank_b],
            )
            os_a, os_b = os_teams[0][0], os_teams[1][0]

            new_ratings = {
                pa["agent_id"]: (ga_after, elo_a_after, os_a),
                pb["agent_id"]: (gb_after, elo_b_after, os_b),
            }
            for p in (pa, pb):
                aid = p["agent_id"]
                ga, elo_v, os_v = new_ratings[aid]
                self._conn.execute(
                    """
                    UPDATE agent_ratings
                    SET glicko_mu = ?, glicko_rd = ?, glicko_vol = ?,
                        elo = ?,
                        os_mu = ?, os_sigma = ?, os_ordinal = ?,
                        matches_played = matches_played + 1,
                        games_played = games_played + ?,
                        wins = wins + ?,
                        losses = losses + ?,
                        draws = draws + ?,
                        p1_wins = p1_wins + ?,
                        p1_games = p1_games + ?,
                        p2_wins = p2_wins + ?,
                        p2_games = p2_games + ?,
                        total_rounds = total_rounds + ?,
                        total_steps = total_steps + ?,
                        source = 'computed',
                        updated_at = ?
                    WHERE agent_id = ?
                    """,
                    (
                        ga.mu,
                        ga.rd,
                        ga.vol,
                        elo_v,
                        os_v.mu,
                        os_v.sigma,
                        os_v.ordinal,
                        p["games"],
                        p["wins"],
                        p["losses"],
                        p["draws"],
                        p["as_p1_wins"],
                        p["as_p1_games"],
                        p["as_p2_wins"],
                        p["as_p2_games"],
                        avg_rounds * p["games"],
                        avg_steps * p["games"],
                        played_at,
                        aid,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO rating_history (
                        project_id, agent_id, run_id, match_id, step,
                        glicko_mu, glicko_rd, elo, os_mu, os_sigma, os_ordinal,
                        source, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'computed', ?)
                    """,
                    (
                        project_id,
                        aid,
                        p.get("run_id") or run_id,
                        match_id,
                        step,
                        ga.mu,
                        ga.rd,
                        elo_v,
                        os_v.mu,
                        os_v.sigma,
                        os_v.ordinal,
                        played_at,
                    ),
                )
                updates.append(
                    {
                        "agent_id": aid,
                        "glicko_mu": ga.mu,
                        "glicko_rd": ga.rd,
                        "elo": elo_v,
                        "os_ordinal": os_v.ordinal,
                    }
                )
        else:
            # Multiplayer FFA or team match via OpenSkill Plackett-Luce
            has_teams = all(bool(p.get("team")) for p in part_rows)
            teams_order: list[str] = []
            teams_members: dict[str, list[dict[str, Any]]] = {}
            for idx, p in enumerate(part_rows):
                tkey = str(p["team"]) if has_teams else f"__solo_{idx}"
                if tkey not in teams_members:
                    teams_order.append(tkey)
                    teams_members[tkey] = []
                teams_members[tkey].append(p)

            if len(teams_order) >= 2:
                os_in = [
                    [
                        OpenSkillRating(
                            mu=float(ratings_by_id[m["agent_id"]]["os_mu"]),
                            sigma=float(ratings_by_id[m["agent_id"]]["os_sigma"]),
                        )
                        for m in teams_members[tk]
                    ]
                    for tk in teams_order
                ]
                team_ranks = [min(int(m["rank"]) for m in teams_members[tk]) for tk in teams_order]
                os_out = rate_openskill_match(os_in, team_ranks)

                for t_idx, tk in enumerate(teams_order):
                    for m_idx, p in enumerate(teams_members[tk]):
                        aid = p["agent_id"]
                        os_v = os_out[t_idx][m_idx]
                        rrow = ratings_by_id[aid]
                        self._conn.execute(
                            """
                            UPDATE agent_ratings
                            SET os_mu = ?, os_sigma = ?, os_ordinal = ?,
                                matches_played = matches_played + 1,
                                games_played = games_played + ?,
                                wins = wins + ?,
                                losses = losses + ?,
                                draws = draws + ?,
                                p1_wins = p1_wins + ?,
                                p1_games = p1_games + ?,
                                p2_wins = p2_wins + ?,
                                p2_games = p2_games + ?,
                                total_rounds = total_rounds + ?,
                                total_steps = total_steps + ?,
                                source = 'computed',
                                updated_at = ?
                            WHERE agent_id = ?
                            """,
                            (
                                os_v.mu,
                                os_v.sigma,
                                os_v.ordinal,
                                p["games"],
                                p["wins"],
                                p["losses"],
                                p["draws"],
                                p["as_p1_wins"],
                                p["as_p1_games"],
                                p["as_p2_wins"],
                                p["as_p2_games"],
                                avg_rounds * p["games"],
                                avg_steps * p["games"],
                                played_at,
                                aid,
                            ),
                        )
                        self._conn.execute(
                            """
                            INSERT INTO rating_history (
                                project_id, agent_id, run_id, match_id, step,
                                glicko_mu, glicko_rd, elo, os_mu, os_sigma, os_ordinal,
                                source, recorded_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'computed', ?)
                            """,
                            (
                                project_id,
                                aid,
                                p.get("run_id") or run_id,
                                match_id,
                                step,
                                float(rrow["glicko_mu"]),
                                float(rrow["glicko_rd"]),
                                float(rrow["elo"]),
                                os_v.mu,
                                os_v.sigma,
                                os_v.ordinal,
                                played_at,
                            ),
                        )
                        updates.append(
                            {
                                "agent_id": aid,
                                "glicko_mu": float(rrow["glicko_mu"]),
                                "glicko_rd": float(rrow["glicko_rd"]),
                                "elo": float(rrow["elo"]),
                                "os_ordinal": os_v.ordinal,
                            }
                        )
            else:
                # Fallback if single-agent
                for aid in distinct_agents:
                    rrow = ratings_by_id[aid]
                    updates.append(
                        {
                            "agent_id": aid,
                            "glicko_mu": float(rrow["glicko_mu"]),
                            "glicko_rd": float(rrow["glicko_rd"]),
                            "elo": float(rrow["elo"]),
                            "os_ordinal": float(rrow["os_ordinal"]),
                        }
                    )
        return updates

    def record_match(self, data: dict[str, Any]) -> dict[str, Any]:
        proj = self.ensure_project(data.get("project") or "default")
        project_id = proj["id"]
        source_key = data.get("source_key")
        played_at = int(data.get("played_at") or now_ms())

        with self._lock:
            if source_key:
                dup = self._conn.execute(
                    "SELECT id FROM matches WHERE project_id = ? AND source_key = ?",
                    (project_id, source_key),
                ).fetchone()
                if dup is not None:
                    return {
                        "match_id": str(dup["id"]),
                        "duplicate": True,
                        "rating_updates": [],
                        "new_best": None,
                    }

            raw_parts = list(data.get("participants") or [])
            if len(raw_parts) < 2:
                raise ValueError("Match requires at least 2 participants")

            total_games = int(data.get("total_games") or 0)
            if total_games <= 0:
                total_games = max(1, sum(int(p.get("games") or 0) for p in raw_parts) // max(1, len(raw_parts)) or 1)
                if len(raw_parts) == 2 and ("wins" in raw_parts[0] or "wins" in raw_parts[1]):
                    total_games = int(raw_parts[0].get("wins", 0)) + int(raw_parts[1].get("wins", 0)) + int(data.get("draws", 0))
                    total_games = max(1, total_games)

            mode = str(data.get("mode") or ("1v1_series" if (len(raw_parts) == 2 and total_games > 1) else "1v1" if len(raw_parts) == 2 else "multiplayer"))
            draws = int(data.get("draws") or 0)
            avg_rounds = float(data.get("avg_rounds") or 0.0)
            avg_steps = float(data.get("avg_steps") or 0.0)
            duration_sec = float(data.get("duration_sec") or 0.0)
            step = int(data["step"]) if data.get("step") is not None else None
            run_id = data.get("run_id")
            if run_id:
                r_exists = self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone()
                if not r_exists:
                    run_id = None

            # Track best non-bot Glicko-2 rating before match
            best_row = self._conn.execute(
                """
                SELECT MAX(ar.glicko_mu) AS max_mu
                FROM agent_ratings ar
                JOIN checkpoints c ON c.id = ar.agent_id
                WHERE ar.project_id = ? AND c.is_bot = 0 AND ar.matches_played > 0
                """,
                (project_id,),
            ).fetchone()
            prev_best_mu = float(best_row["max_mu"]) if best_row and best_row["max_mu"] is not None else None

            # Resolve or auto-create checkpoint rows for each participant
            normalized_parts: list[dict[str, Any]] = []
            for idx, p in enumerate(raw_parts):
                agent_ref = str(p.get("agent") or p.get("agent_id") or p.get("name"))
                ckpt = self.get_checkpoint(agent_ref) or self.get_checkpoint_by_name(project_id, agent_ref)
                if ckpt is None:
                    is_bot = self.is_bot_checkpoint(agent_ref)
                    ckpt = self.upsert_checkpoint(
                        {
                            "project": project_id,
                            "name": agent_ref,
                            "storage_mode": "builtin" if is_bot else "path_ref",
                            "path_or_blob": f"builtin:{agent_ref}" if is_bot else "",
                            "is_bot": is_bot,
                        }
                    )
                p_run_id = p.get("run_id") or ckpt.get("run_id") or run_id
                if p_run_id:
                    if not self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (p_run_id,)).fetchone():
                        p_run_id = None

                games = int(p.get("games") or total_games)
                wins = int(p.get("wins") if p.get("wins") is not None else (1 if p.get("won") or p.get("rank") == 1 else 0))
                p_draws = int(p.get("draws") if p.get("draws") is not None else draws)
                losses = int(p.get("losses") if p.get("losses") is not None else max(0, games - wins - p_draws))
                seat = int(p["seat"]) if p.get("seat") is not None else idx
                rank = int(p.get("rank") if p.get("rank") is not None else (1 if wins > losses else 2))

                as_p1_wins = int(p.get("as_p1_wins") or (wins if seat == 0 and "as_p1_games" not in p else 0))
                as_p1_games = int(p.get("as_p1_games") or (games if seat == 0 and "as_p1_games" not in p else 0))
                as_p2_wins = int(p.get("as_p2_wins") or (wins if seat == 1 and "as_p2_games" not in p else 0))
                as_p2_games = int(p.get("as_p2_games") or (games if seat == 1 and "as_p2_games" not in p else 0))

                normalized_parts.append(
                    {
                        "agent_id": ckpt["id"],
                        "agent_name": ckpt["name"],
                        "is_bot": ckpt["is_bot"],
                        "run_id": p_run_id,
                        "seat": seat,
                        "team": p.get("team"),
                        "deck_or_faction": p.get("deck_or_faction") or p.get("deck"),
                        "rank": rank,
                        "score": float(p.get("score") if p.get("score") is not None else wins),
                        "wins": wins,
                        "losses": losses,
                        "draws": p_draws,
                        "games": games,
                        "as_p1_wins": as_p1_wins,
                        "as_p1_games": as_p1_games,
                        "as_p2_wins": as_p2_wins,
                        "as_p2_games": as_p2_games,
                    }
                )

            p1_side_wins = int(data.get("p1_side_wins") if data.get("p1_side_wins") is not None else sum(p["as_p1_wins"] for p in normalized_parts))
            p2_side_wins = int(data.get("p2_side_wins") if data.get("p2_side_wins") is not None else sum(p["as_p2_wins"] for p in normalized_parts))

            match_id = f"match_{uuid.uuid4().hex[:12]}"
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO matches (
                        id, project_id, run_id, source_key, step, mode, total_games,
                        draws, p1_side_wins, p2_side_wins, avg_rounds, avg_steps,
                        duration_sec, manifest_hash, replay_id, metadata_json, played_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        project_id,
                        run_id,
                        source_key,
                        step,
                        mode,
                        total_games,
                        draws,
                        p1_side_wins,
                        p2_side_wins,
                        avg_rounds,
                        avg_steps,
                        duration_sec,
                        data.get("manifest_hash"),
                        data.get("replay_id"),
                        _json_dumps(sanitize_json_value(data.get("metadata") or {})),
                        played_at,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO match_participants (
                        match_id, agent_id, run_id, seat, team, deck_or_faction, rank,
                        score, wins, losses, draws, games, as_p1_wins, as_p1_games,
                        as_p2_wins, as_p2_games
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            match_id,
                            p["agent_id"],
                            p["run_id"],
                            p["seat"],
                            p["team"],
                            p["deck_or_faction"],
                            p["rank"],
                            p["score"],
                            p["wins"],
                            p["losses"],
                            p["draws"],
                            p["games"],
                            p["as_p1_wins"],
                            p["as_p1_games"],
                            p["as_p2_wins"],
                            p["as_p2_games"],
                        )
                        for p in normalized_parts
                    ],
                )
                rating_updates = self._apply_match_ratings_locked(
                    project_id=project_id,
                    match_id=match_id,
                    mode=mode,
                    total_games=total_games,
                    draws=draws,
                    avg_rounds=avg_rounds,
                    avg_steps=avg_steps,
                    step=step,
                    run_id=run_id,
                    played_at=played_at,
                    part_rows=normalized_parts,
                )

            new_best = None
            non_bot_by_id = {p["agent_id"]: p for p in normalized_parts if not p["is_bot"]}
            best_candidate = None
            for upd in rating_updates:
                if upd["agent_id"] in non_bot_by_id:
                    mu = upd["glicko_mu"]
                    if prev_best_mu is None:
                        if mu > 1500.0 and (best_candidate is None or mu > best_candidate["glicko_mu"]):
                            best_candidate = upd
                    elif mu > prev_best_mu and (best_candidate is None or mu > best_candidate["glicko_mu"]):
                        best_candidate = upd
            if best_candidate is not None:
                ckpt_best = self.get_checkpoint(best_candidate["agent_id"])
                if ckpt_best is not None:
                    new_best = {
                        "agent": ckpt_best,
                        "glicko_mu": best_candidate["glicko_mu"],
                        "glicko_rd": best_candidate["glicko_rd"],
                    }

            return {
                "match_id": match_id,
                "project_id": project_id,
                "duplicate": False,
                "rating_updates": rating_updates,
                "new_best": new_best,
            }

    def log_direct_rating(self, data: dict[str, Any]) -> dict[str, Any]:
        proj = self.ensure_project(data.get("project") or "default")
        project_id = proj["id"]
        agent_ref = str(data["agent"])
        ts = int(data.get("recorded_at") or now_ms())
        step = int(data["step"]) if data.get("step") is not None else None
        run_id = data.get("run_id")

        with self._lock:
            ckpt = self.get_checkpoint(agent_ref) or self.get_checkpoint_by_name(project_id, agent_ref)
            if ckpt is None:
                ckpt = self.upsert_checkpoint({"project": project_id, "name": agent_ref, "run_id": run_id})
            aid = ckpt["id"]
            self._ensure_rating_row_locked(aid, project_id, ts)
            row = self._conn.execute("SELECT * FROM agent_ratings WHERE agent_id = ?", (aid,)).fetchone()
            assert row is not None

            glicko_mu = float(data["glicko_mu"]) if data.get("glicko_mu") is not None else float(row["glicko_mu"])
            glicko_rd = float(data["glicko_rd"]) if data.get("glicko_rd") is not None else float(row["glicko_rd"])
            elo = float(data["elo"]) if data.get("elo") is not None else float(row["elo"])
            os_mu = float(data["os_mu"]) if data.get("os_mu") is not None else float(row["os_mu"])
            os_sigma = float(data["os_sigma"]) if data.get("os_sigma") is not None else float(row["os_sigma"])
            os_ordinal = os_mu - 3.0 * os_sigma

            if run_id and not self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
                run_id = None

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE agent_ratings
                    SET glicko_mu = ?, glicko_rd = ?, elo = ?,
                        os_mu = ?, os_sigma = ?, os_ordinal = ?,
                        source = 'logged', updated_at = ?
                    WHERE agent_id = ?
                    """,
                    (glicko_mu, glicko_rd, elo, os_mu, os_sigma, os_ordinal, ts, aid),
                )
                self._conn.execute(
                    """
                    INSERT INTO rating_history (
                        project_id, agent_id, run_id, match_id, step,
                        glicko_mu, glicko_rd, elo, os_mu, os_sigma, os_ordinal,
                        source, recorded_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 'logged', ?)
                    """,
                    (project_id, aid, run_id, step, glicko_mu, glicko_rd, elo, os_mu, os_sigma, os_ordinal, ts),
                )
            return {
                "agent_id": aid,
                "name": ckpt["name"],
                "glicko_mu": glicko_mu,
                "glicko_rd": glicko_rd,
                "elo": elo,
                "os_mu": os_mu,
                "os_sigma": os_sigma,
                "os_ordinal": os_ordinal,
                "source": "logged",
            }

    def recompute_project_ratings(self, project: str) -> dict[str, Any]:
        proj_id = self._resolve_project_id(project)
        if not proj_id:
            return {"recomputed_matches": 0, "agents": 0}
        ts = now_ms()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE agent_ratings
                    SET glicko_mu = 1500.0, glicko_rd = 350.0, glicko_vol = 0.06,
                        elo = 1500.0,
                        os_mu = 25.0, os_sigma = 8.333333333333334, os_ordinal = 0.0,
                        matches_played = 0, games_played = 0,
                        wins = 0, losses = 0, draws = 0,
                        p1_wins = 0, p1_games = 0, p2_wins = 0, p2_games = 0,
                        total_rounds = 0.0, total_steps = 0.0,
                        source = 'computed', updated_at = ?
                    WHERE project_id = ?
                    """,
                    (ts, proj_id),
                )
                self._conn.execute(
                    "DELETE FROM rating_history WHERE project_id = ? AND source = 'computed'",
                    (proj_id,),
                )
                match_rows = self._conn.execute(
                    "SELECT * FROM matches WHERE project_id = ? ORDER BY played_at ASC, id ASC",
                    (proj_id,),
                ).fetchall()
                for mr in match_rows:
                    parts = self._conn.execute(
                        "SELECT * FROM match_participants WHERE match_id = ? ORDER BY id ASC",
                        (mr["id"],),
                    ).fetchall()
                    part_dicts = [dict(p) for p in parts]
                    self._apply_match_ratings_locked(
                        project_id=proj_id,
                        match_id=mr["id"],
                        mode=mr["mode"],
                        total_games=int(mr["total_games"]),
                        draws=int(mr["draws"]),
                        avg_rounds=float(mr["avg_rounds"]),
                        avg_steps=float(mr["avg_steps"]),
                        step=mr["step"],
                        run_id=mr["run_id"],
                        played_at=int(mr["played_at"]),
                        part_rows=part_dicts,
                    )
            cnt = self._conn.execute(
                "SELECT COUNT(*) AS c FROM agent_ratings WHERE project_id = ?", (proj_id,)
            ).fetchone()
            return {
                "recomputed_matches": len(match_rows),
                "agents": int(cnt["c"]) if cnt else 0,
            }

    def get_leaderboard(
        self,
        *,
        project: str | None = None,
        arch: str | None = None,
        manifest_hash: str | None = None,
        saved_only: bool = False,
        include_bots: bool = True,
        search: str | None = None,
        sort_by: str = "glicko_mu",
        include_unplayed: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if arch:
            clauses.append("c.arch = ?")
            params.append(arch)
        if manifest_hash:
            clauses.append("c.manifest_hash = ?")
            params.append(manifest_hash)
        if not include_bots:
            clauses.append("c.is_bot = 0")
        if not include_unplayed:
            clauses.append("(ar.games_played > 0 OR ar.source = 'logged')")
        if search:
            clauses.append("(c.name LIKE ? OR COALESCE(c.arch, '') LIKE ?)")
            pat = f"%{search}%"
            params.extend([pat, pat])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.*, p.name AS project_name,
                       ar.glicko_mu, ar.glicko_rd, ar.glicko_vol, ar.elo,
                       ar.os_mu, ar.os_sigma, ar.os_ordinal,
                       ar.matches_played, ar.games_played,
                       ar.wins, ar.losses, ar.draws,
                       ar.p1_wins, ar.p1_games, ar.p2_wins, ar.p2_games,
                       ar.total_rounds, ar.total_steps, ar.source, ar.updated_at
                FROM checkpoints c
                JOIN projects p ON p.id = c.project_id
                JOIN agent_ratings ar ON ar.agent_id = c.id
                {where}
                """,
                params,
            ).fetchall()

        entries: list[dict[str, Any]] = []
        for row in rows:
            d = self._hydrate_checkpoint(row)
            if saved_only and not self.is_saved_or_named_checkpoint(d):
                continue
            gp = max(int(d.get("games_played") or 0), 1)
            raw_gp = int(d.get("games_played") or 0)
            d["win_rate"] = (d["wins"] / gp) if raw_gp > 0 else 0.0
            d["p1_win_rate"] = (d["p1_wins"] / d["p1_games"]) if d["p1_games"] > 0 else None
            d["p2_win_rate"] = (d["p2_wins"] / d["p2_games"]) if d["p2_games"] > 0 else None
            d["avg_rounds"] = (d["total_rounds"] / gp) if raw_gp > 0 else 0.0
            d["avg_steps"] = (d["total_steps"] / gp) if raw_gp > 0 else 0.0
            d["ci_95"] = round(2.0 * float(d["glicko_rd"]), 2)
            entries.append(d)

        sort_key = {
            "glicko_mu": lambda x: float(x["glicko_mu"]),
            "elo": lambda x: float(x["elo"]),
            "os_ordinal": lambda x: float(x["os_ordinal"]),
            "win_rate": lambda x: float(x["win_rate"]),
            "games_played": lambda x: int(x["games_played"]),
        }.get(sort_by, lambda x: float(x["glicko_mu"]))
        entries.sort(key=sort_key, reverse=True)
        for idx, e in enumerate(entries, start=1):
            e["rank"] = idx
        return entries

    def get_h2h_matrix(
        self,
        *,
        project: str | None = None,
        arch: str | None = None,
        manifest_hash: str | None = None,
        saved_only: bool = False,
        include_bots: bool = True,
        search: str | None = None,
    ) -> dict[str, Any]:
        agents = self.get_leaderboard(
            project=project,
            arch=arch,
            manifest_hash=manifest_hash,
            saved_only=saved_only,
            include_bots=include_bots,
            search=search,
            sort_by="glicko_mu",
        )
        agent_ids = [a["id"] for a in agents]
        agent_set = set(agent_ids)
        cells: dict[str, dict[str, dict[str, Any]]] = {aid: {} for aid in agent_ids}

        if len(agent_ids) < 2:
            return {"agents": agents, "cells": cells}

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT mp1.agent_id AS agent_a,
                       mp2.agent_id AS agent_b,
                       SUM(mp1.wins) AS a_wins,
                       SUM(mp2.wins) AS b_wins,
                       SUM(m.draws) AS draws,
                       SUM(m.total_games) AS total_games,
                       SUM(mp1.as_p1_wins) AS a_p1_wins,
                       SUM(mp1.as_p1_games) AS a_p1_games,
                       SUM(mp1.as_p2_wins) AS a_p2_wins,
                       SUM(mp1.as_p2_games) AS a_p2_games,
                       COUNT(*) AS match_count
                FROM match_participants mp1
                JOIN match_participants mp2 ON mp1.match_id = mp2.match_id AND mp1.agent_id != mp2.agent_id
                JOIN matches m ON m.id = mp1.match_id
                WHERE m.mode IN ('1v1', '1v1_series')
                GROUP BY mp1.agent_id, mp2.agent_id
                """
            ).fetchall()

        for r in rows:
            a = str(r["agent_a"])
            b = str(r["agent_b"])
            if a not in agent_set or b not in agent_set:
                continue
            tot = int(r["total_games"] or 0)
            if tot <= 0:
                continue
            aw = int(r["a_wins"] or 0)
            bw = int(r["b_wins"] or 0)
            dr = int(r["draws"] or 0)
            p1g = int(r["a_p1_games"] or 0)
            p2g = int(r["a_p2_games"] or 0)
            cells[a][b] = {
                "wins": aw,
                "losses": bw,
                "draws": dr,
                "total_games": tot,
                "win_rate": (aw + 0.5 * dr) / tot,
                "as_p1_wr": (int(r["a_p1_wins"] or 0) / p1g) if p1g > 0 else None,
                "as_p2_wr": (int(r["a_p2_wins"] or 0) / p2g) if p2g > 0 else None,
                "match_count": int(r["match_count"] or 0),
            }
        return {"agents": agents, "cells": cells}

    def get_seat_and_deck_stats(self, project: str | None = None) -> dict[str, Any]:
        proj_clause = ""
        params: list[Any] = []
        if project:
            proj_clause = "WHERE (p.name = ? OR p.id = ?)"
            params.extend([project, project])

        with self._lock:
            side_row = self._conn.execute(
                f"""
                SELECT COALESCE(SUM(m.p1_side_wins), 0) AS p1_wins,
                       COALESCE(SUM(m.p2_side_wins), 0) AS p2_wins,
                       COALESCE(SUM(m.draws), 0) AS draws,
                       COALESCE(SUM(m.total_games), 0) AS total_games
                FROM matches m
                JOIN projects p ON p.id = m.project_id
                {proj_clause}
                """,
                params,
            ).fetchone()

            seat_rows = self._conn.execute(
                f"""
                SELECT mp.seat,
                       SUM(mp.wins) AS wins,
                       SUM(mp.losses) AS losses,
                       SUM(mp.draws) AS draws,
                       SUM(mp.games) AS games
                FROM match_participants mp
                JOIN matches m ON m.id = mp.match_id
                JOIN projects p ON p.id = m.project_id
                {proj_clause}
                GROUP BY mp.seat
                ORDER BY mp.seat ASC
                """,
                params,
            ).fetchall()

            deck_rows = self._conn.execute(
                f"""
                SELECT mp1.deck_or_faction AS deck_a,
                       mp2.deck_or_faction AS deck_b,
                       SUM(mp1.wins) AS a_wins,
                       SUM(mp2.wins) AS b_wins,
                       SUM(m.draws) AS draws,
                       SUM(m.total_games) AS total_games
                FROM match_participants mp1
                JOIN match_participants mp2 ON mp1.match_id = mp2.match_id AND mp1.id < mp2.id
                JOIN matches m ON m.id = mp1.match_id
                JOIN projects p ON p.id = m.project_id
                {proj_clause}
                {"AND" if proj_clause else "WHERE"} mp1.deck_or_faction IS NOT NULL
                  AND mp2.deck_or_faction IS NOT NULL
                GROUP BY mp1.deck_or_faction, mp2.deck_or_faction
                ORDER BY total_games DESC
                """,
                params,
            ).fetchall()

        p1_w = int(side_row["p1_wins"] or 0) if side_row else 0
        p2_w = int(side_row["p2_wins"] or 0) if side_row else 0
        tot_side = max(p1_w + p2_w, 1)
        return {
            "side_split": {
                "p1_wins": p1_w,
                "p2_wins": p2_w,
                "draws": int(side_row["draws"] or 0) if side_row else 0,
                "total_games": int(side_row["total_games"] or 0) if side_row else 0,
                "p1_win_rate": p1_w / tot_side if (p1_w + p2_w) > 0 else 0.5,
                "p2_win_rate": p2_w / tot_side if (p1_w + p2_w) > 0 else 0.5,
            },
            "seats": [
                {
                    "seat": r["seat"],
                    "wins": int(r["wins"] or 0),
                    "losses": int(r["losses"] or 0),
                    "draws": int(r["draws"] or 0),
                    "games": int(r["games"] or 0),
                    "win_rate": (int(r["wins"] or 0) / max(int(r["games"] or 0), 1)),
                }
                for r in seat_rows
                if r["seat"] is not None
            ],
            "deck_matchups": [
                {
                    "deck_a": r["deck_a"],
                    "deck_b": r["deck_b"],
                    "a_wins": int(r["a_wins"] or 0),
                    "b_wins": int(r["b_wins"] or 0),
                    "draws": int(r["draws"] or 0),
                    "total_games": int(r["total_games"] or 0),
                    "a_win_rate": (int(r["a_wins"] or 0) + 0.5 * int(r["draws"] or 0))
                    / max(int(r["total_games"] or 0), 1),
                }
                for r in deck_rows
            ],
        }

    def list_matches(
        self,
        *,
        project: str | None = None,
        agent_id: str | None = None,
        opponent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if agent_id:
            clauses.append("EXISTS (SELECT 1 FROM match_participants mp WHERE mp.match_id = m.id AND mp.agent_id = ?)")
            params.append(agent_id)
        if opponent_id:
            clauses.append("EXISTS (SELECT 1 FROM match_participants mp WHERE mp.match_id = m.id AND mp.agent_id = ?)")
            params.append(opponent_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            m_rows = self._conn.execute(
                f"""
                SELECT m.*, p.name AS project_name
                FROM matches m
                JOIN projects p ON p.id = m.project_id
                {where}
                ORDER BY m.played_at DESC, m.id DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for mr in m_rows:
                d = dict(mr)
                d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
                parts = self._conn.execute(
                    """
                    SELECT mp.*, c.name AS agent_name, c.arch AS arch
                    FROM match_participants mp
                    JOIN checkpoints c ON c.id = mp.agent_id
                    WHERE mp.match_id = ?
                    ORDER BY mp.seat ASC, mp.id ASC
                    """,
                    (d["id"],),
                ).fetchall()
                d["participants"] = [dict(p) for p in parts]
                out.append(d)
            return out

    def get_rating_history(
        self,
        *,
        project: str | None = None,
        agent_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            clauses.append(f"rh.agent_id IN ({placeholders})")
            params.extend(agent_ids)
        if run_id:
            clauses.append("rh.run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT rh.*, c.name AS agent_name
                FROM rating_history rh
                JOIN checkpoints c ON c.id = rh.agent_id
                JOIN projects p ON p.id = rh.project_id
                {where}
                ORDER BY rh.recorded_at ASC, rh.id ASC
                """,
                params,
            ).fetchall()
            grouped: dict[str, list[dict[str, Any]]] = {}
            for r in rows:
                d = dict(r)
                grouped.setdefault(d["agent_id"], []).append(d)
            return grouped

    # ------------------------------------------------------------------
    # ReDo Neuron Dormancy & Effective Rank (Plasticity Reports)
    # ------------------------------------------------------------------

    def insert_plasticity_report(self, data: dict[str, Any]) -> dict[str, Any]:
        proj = self.ensure_project(data.get("project") or "default")
        project_id = proj["id"]
        layers = list(data.get("layers") or [])
        meta = dict(data.get("meta") or {})
        model_name = str(data.get("model_name") or meta.get("model_name") or data.get("checkpoint") or "model")
        arch = data.get("arch") or meta.get("arch")
        hook_mode = str(data.get("hook_mode") or meta.get("hook_mode") or "post_activation")
        total_obs = int(data.get("total_obs") or meta.get("total_obs") or 0)
        step = data.get("step")
        if step is None and meta.get("step") is not None:
            step = int(meta["step"])

        normalized_layers: list[dict[str, Any]] = []
        for lyr in layers:
            dim = int(lyr.get("dim") or 0)
            hd_c = int(lyr.get("hard_dead_count") or 0)
            dm_c = int(lyr.get("dormant_count") or 0)
            wk_c = int(lyr.get("weak_count") or 0)
            rank_ratio = lyr.get("rank_ratio_pct")
            if rank_ratio is None:
                rank_ratio = lyr.get("rank_ratio")
            if rank_ratio is None:
                erank = float(lyr.get("effective_rank") or 0.0)
                rank_ratio = (erank / max(dim, 1)) * 100.0
            normalized_layers.append(
                {
                    "name": str(lyr.get("name") or ""),
                    "hook_type": str(lyr.get("hook_type") or "Linear"),
                    "dim": dim,
                    "hard_dead_count": hd_c,
                    "hard_dead_pct": float(lyr.get("hard_dead_pct") if lyr.get("hard_dead_pct") is not None else (hd_c / max(dim, 1)) * 100.0),
                    "dormant_count": dm_c,
                    "dormant_pct": float(lyr.get("dormant_pct") if lyr.get("dormant_pct") is not None else (dm_c / max(dim, 1)) * 100.0),
                    "weak_count": wk_c,
                    "weak_pct": float(lyr.get("weak_pct") if lyr.get("weak_pct") is not None else (wk_c / max(dim, 1)) * 100.0),
                    "effective_rank": float(lyr.get("effective_rank") or 0.0),
                    "rank_ratio_pct": float(rank_ratio),
                }
            )

        total_neurons = int(data.get("total_neurons") or sum(l["dim"] for l in normalized_layers))
        hd_total = int(data.get("hard_dead_count") or sum(l["hard_dead_count"] for l in normalized_layers))
        dm_total = int(data.get("dormant_count") or sum(l["dormant_count"] for l in normalized_layers))
        wk_total = int(data.get("weak_count") or sum(l["weak_count"] for l in normalized_layers))
        hd_pct = float(data.get("hard_dead_pct") if data.get("hard_dead_pct") is not None else (hd_total / max(total_neurons, 1)) * 100.0)
        dm_pct = float(data.get("dormant_pct") if data.get("dormant_pct") is not None else (dm_total / max(total_neurons, 1)) * 100.0)
        wk_pct = float(data.get("weak_pct") if data.get("weak_pct") is not None else (wk_total / max(total_neurons, 1)) * 100.0)
        mean_rank_pct = float(
            data.get("mean_rank_ratio_pct")
            if data.get("mean_rank_ratio_pct") is not None
            else (sum(l["rank_ratio_pct"] for l in normalized_layers) / max(len(normalized_layers), 1))
        )

        ts = int(data.get("created_at") or now_ms())
        rep_id = f"plast_{uuid.uuid4().hex[:12]}"

        with self._lock:
            run_id = data.get("run_id")
            if run_id and not self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
                run_id = None
            ckpt_id = self._resolve_checkpoint_ref_locked(
                project_id, data.get("checkpoint_id") or data.get("checkpoint") or model_name
            )
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO plasticity_reports (
                        id, project_id, run_id, checkpoint_id, model_name, step, arch,
                        hook_mode, total_obs, total_neurons, hard_dead_count, hard_dead_pct,
                        dormant_count, dormant_pct, weak_count, weak_pct,
                        mean_rank_ratio_pct, layers_json, meta_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rep_id,
                        project_id,
                        run_id,
                        ckpt_id,
                        model_name,
                        step,
                        arch,
                        hook_mode,
                        total_obs,
                        total_neurons,
                        hd_total,
                        hd_pct,
                        dm_total,
                        dm_pct,
                        wk_total,
                        wk_pct,
                        mean_rank_pct,
                        _json_dumps(sanitize_json_value(normalized_layers)),
                        _json_dumps(sanitize_json_value(meta)),
                        ts,
                    ),
                )
            res = self.get_plasticity_report(rep_id)
            assert res is not None
            return res

    def _hydrate_plasticity(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["layers"] = json.loads(d.pop("layers_json", "[]") or "[]")
        d["meta"] = json.loads(d.pop("meta_json", "{}") or "{}")
        return d

    def get_plasticity_report(self, report_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT pr.*, p.name AS project_name, c.name AS checkpoint_name
                FROM plasticity_reports pr
                JOIN projects p ON p.id = pr.project_id
                LEFT JOIN checkpoints c ON c.id = pr.checkpoint_id
                WHERE pr.id = ?
                """,
                (report_id,),
            ).fetchone()
            return self._hydrate_plasticity(row) if row else None

    def list_plasticity_reports(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        checkpoint_id: str | None = None,
        model_name: str | None = None,
        arch: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("pr.run_id = ?")
            params.append(run_id)
        if checkpoint_id:
            clauses.append("pr.checkpoint_id = ?")
            params.append(checkpoint_id)
        if model_name:
            clauses.append("pr.model_name = ?")
            params.append(model_name)
        if arch:
            clauses.append("pr.arch = ?")
            params.append(arch)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT pr.*, p.name AS project_name, c.name AS checkpoint_name
                FROM plasticity_reports pr
                JOIN projects p ON p.id = pr.project_id
                LEFT JOIN checkpoints c ON c.id = pr.checkpoint_id
                {where}
                ORDER BY pr.created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            return [self._hydrate_plasticity(r) for r in rows]

    # ------------------------------------------------------------------
    # Replays & RL Traces
    # ------------------------------------------------------------------

    def _hydrate_replay(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["deck1"] = json.loads(d.pop("deck1_json", "[]") or "[]")
        d["deck2"] = json.loads(d.pop("deck2_json", "[]") or "[]")
        d["config"] = json.loads(d.pop("config_json", "{}") or "{}")
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
        d["meta"] = json.loads(d.pop("meta_json", "{}") or "{}")
        d["has_rl_trace"] = bool(d.get("has_rl_trace"))
        d["pinned"] = bool(d.get("pinned"))
        return sanitize_json_value(d)

    def insert_replay(self, data: dict[str, Any]) -> dict[str, Any]:
        import math

        proj = self.ensure_project(data.get("project") or "default")
        project_id = proj["id"]
        replay_id = str(data.get("id") or f"rpl_{uuid.uuid4().hex[:12]}")
        ts = int(data.get("created_at") or now_ms())

        swing_raw = data.get("max_value_swing")
        try:
            swing = float(swing_raw) if swing_raw is not None else 0.0
        except (TypeError, ValueError):
            swing = 0.0
        if not math.isfinite(swing):
            swing = 0.0

        tags = list(dict.fromkeys(str(t) for t in (data.get("tags") or []) if t))
        deck1 = [str(c) for c in (data.get("deck1") or [])]
        deck2 = [str(c) for c in (data.get("deck2") or [])]
        config = sanitize_json_value(data.get("config") or {})
        meta = sanitize_json_value(data.get("meta") or {})

        with self._lock:
            run_id = data.get("run_id")
            if run_id:
                if not self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
                    run_id = None
            match_id = data.get("match_id")
            if match_id:
                if not self._conn.execute("SELECT 1 FROM matches WHERE id = ?", (match_id,)).fetchone():
                    match_id = None

            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO replays (
                        id, project_id, run_id, match_id, title, format, content_type,
                        blob_sha256, size_bytes, filename, step, seed1, seed2, version,
                        manifest_hash, deck1_json, deck2_json, config_json, action_count,
                        winner, p1_name, p2_name, has_rl_trace, trace_sha256,
                        max_value_swing, blunder_turn, tags_json, pinned, meta_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        replay_id,
                        project_id,
                        run_id,
                        match_id,
                        str(data.get("title") or data.get("filename") or replay_id),
                        str(data.get("format") or "bin"),
                        str(data.get("content_type") or "application/octet-stream"),
                        str(data["blob_sha256"]),
                        int(data.get("size_bytes") or 0),
                        str(data.get("filename") or ""),
                        int(data["step"]) if data.get("step") is not None else None,
                        int(data["seed1"]) if data.get("seed1") is not None else None,
                        int(data["seed2"]) if data.get("seed2") is not None else None,
                        int(data["version"]) if data.get("version") is not None else None,
                        str(data.get("manifest_hash") or ""),
                        _json_dumps(deck1),
                        _json_dumps(deck2),
                        _json_dumps(config),
                        int(data.get("action_count") or 0),
                        int(data["winner"]) if data.get("winner") is not None else None,
                        str(data.get("p1_name") or ""),
                        str(data.get("p2_name") or ""),
                        1 if data.get("has_rl_trace") else 0,
                        data.get("trace_sha256"),
                        swing,
                        int(data["blunder_turn"]) if data.get("blunder_turn") is not None else None,
                        _json_dumps(tags),
                        1 if data.get("pinned") else 0,
                        _json_dumps(meta),
                        ts,
                    ),
                )
            res = self.get_replay(replay_id)
            assert res is not None
            return res

    def get_replay(self, replay_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT rp.*, p.name AS project_name, r.name AS run_name
                FROM replays rp
                JOIN projects p ON p.id = rp.project_id
                LEFT JOIN runs r ON r.id = rp.run_id
                WHERE rp.id = ?
                """,
                (replay_id,),
            ).fetchone()
            return self._hydrate_replay(row) if row else None

    def list_replays(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        match_id: str | None = None,
        format: str | None = None,
        deck: str | None = None,
        tag: str | None = None,
        has_trace: bool | None = None,
        min_swing: float | None = None,
        pinned_only: bool = False,
        q: str | None = None,
        sort_by: str = "created_at",
        order: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("rp.run_id = ?")
            params.append(run_id)
        if match_id:
            clauses.append("rp.match_id = ?")
            params.append(match_id)
        if format:
            clauses.append("rp.format = ?")
            params.append(format)
        if deck:
            clauses.append("(INSTR(LOWER(rp.deck1_json), LOWER(?)) > 0 OR INSTR(LOWER(rp.deck2_json), LOWER(?)) > 0)")
            params.extend([deck, deck])
        if tag:
            clauses.append("INSTR(rp.tags_json, ?) > 0")
            params.append(f'"{tag}"')
        if has_trace is True:
            clauses.append("rp.has_rl_trace = 1")
        elif has_trace is False:
            clauses.append("rp.has_rl_trace = 0")
        if min_swing is not None:
            clauses.append("rp.max_value_swing >= ?")
            params.append(float(min_swing))
        if pinned_only:
            clauses.append("rp.pinned = 1")
        if q:
            clauses.append(
                "(INSTR(LOWER(rp.title), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.p1_name), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.p2_name), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.manifest_hash), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.filename), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.deck1_json), LOWER(?)) > 0 "
                "OR INSTR(LOWER(rp.deck2_json), LOWER(?)) > 0)"
            )
            params.extend([q] * 7)

        sort_map = {
            "created_at": "rp.created_at",
            "max_value_swing": "rp.max_value_swing",
            "step": "COALESCE(rp.step, -1)",
            "action_count": "rp.action_count",
        }
        order_col = sort_map.get(sort_by, "rp.created_at")
        order_dir = "ASC" if str(order).lower() == "asc" else "DESC"
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT rp.*, p.name AS project_name, r.name AS run_name
                FROM replays rp
                JOIN projects p ON p.id = rp.project_id
                LEFT JOIN runs r ON r.id = rp.run_id
                {where}
                ORDER BY rp.pinned DESC, {order_col} {order_dir}, rp.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            out = [self._hydrate_replay(r) for r in rows]
            if tag:
                out = [r for r in out if tag in r.get("tags", [])]
            return out

    def update_replay(
        self,
        replay_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM replays WHERE id = ?", (replay_id,)).fetchone()
            if row is None:
                return None
            sets: list[str] = []
            params: list[Any] = []
            if title is not None:
                sets.append("title = ?")
                params.append(title)
            if pinned is not None:
                sets.append("pinned = ?")
                params.append(1 if pinned else 0)
            if tags is not None:
                clean_tags = list(dict.fromkeys(str(t) for t in tags if t))
                sets.append("tags_json = ?")
                params.append(_json_dumps(clean_tags))
            if sets:
                params.append(replay_id)
                with self._conn:
                    self._conn.execute(
                        f"UPDATE replays SET {', '.join(sets)} WHERE id = ?",
                        tuple(params),
                    )
            return self.get_replay(replay_id)

    def delete_replay(self, replay_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self.get_replay(replay_id)
            if existing is None:
                return None
            with self._conn:
                self._conn.execute("DELETE FROM replays WHERE id = ?", (replay_id,))
            return existing

    def count_replay_blob_refs(self, *, blob_sha256: str | None = None, trace_sha256: str | None = None) -> dict[str, int]:
        with self._lock:
            blob_refs = 0
            trace_refs = 0
            if blob_sha256:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM replays WHERE blob_sha256 = ?",
                    (blob_sha256,),
                ).fetchone()
                blob_refs = int(row["n"]) if row else 0
            if trace_sha256:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM replays WHERE trace_sha256 = ?",
                    (trace_sha256,),
                ).fetchone()
                trace_refs = int(row["n"]) if row else 0
            return {"blob_refs": blob_refs, "trace_refs": trace_refs}

    # ------------------------------------------------------------------
    # Card Meta Analytics & Distributions
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_opt_float(val: Any) -> float | None:
        import math

        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def insert_card_stats(
        self,
        *,
        project: str,
        run_id: str | None,
        step: int,
        items: list[dict[str, Any]],
        recorded_at: int | None = None,
    ) -> dict[str, Any]:
        proj = self.ensure_project(project or "default")
        project_id = proj["id"]
        ts = int(recorded_at or now_ms())

        with self._lock:
            eff_run_id = run_id
            if eff_run_id:
                if not self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (eff_run_id,)).fetchone():
                    eff_run_id = None

            rows_to_insert: list[tuple[Any, ...]] = []
            for item in items:
                card_name = str(item.get("card_name") or item.get("card") or "").strip()
                if not card_name:
                    continue
                played = int(item.get("played") or 0)
                drawn = int(item.get("drawn") or 0)
                wins_wp = int(item.get("wins_when_played") or 0)
                games_wp = int(item.get("games_when_played") or 0)
                play_rate = self._clean_opt_float(item.get("play_rate"))
                win_rate = self._clean_opt_float(item.get("win_rate"))
                if win_rate is None and games_wp > 0:
                    win_rate = wins_wp / games_wp
                if play_rate is None and drawn > 0:
                    play_rate = played / drawn
                extra = sanitize_json_value(item.get("extra") or {})
                rows_to_insert.append(
                    (
                        project_id,
                        eff_run_id,
                        int(step),
                        card_name,
                        played,
                        drawn,
                        wins_wp,
                        games_wp,
                        play_rate,
                        win_rate,
                        _json_dumps(extra),
                        ts,
                    )
                )

            if rows_to_insert:
                with self._conn:
                    self._conn.executemany(
                        """
                        INSERT INTO card_stats (
                            project_id, run_id, step, card_name,
                            played, drawn, wins_when_played, games_when_played,
                            play_rate, win_rate, extra_json, recorded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows_to_insert,
                    )
            return {
                "project": proj["name"],
                "project_id": project_id,
                "run_id": eff_run_id,
                "step": int(step),
                "inserted": len(rows_to_insert),
            }

    def get_card_meta(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        min_played: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("cs.run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT cs.card_name,
                       cs.step,
                       cs.played,
                       cs.drawn,
                       cs.wins_when_played,
                       cs.games_when_played,
                       cs.play_rate,
                       cs.win_rate,
                       cs.recorded_at
                FROM card_stats cs
                JOIN projects p ON p.id = cs.project_id
                {where}
                ORDER BY cs.step ASC, cs.id ASC
                """,
                tuple(params),
            ).fetchall()

        if not rows:
            return []

        by_card: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_card.setdefault(str(r["card_name"]), []).append(r)

        total_played_all = sum(int(r["played"] or 0) for r in rows)
        out: list[dict[str, Any]] = []
        for card_name, card_rows in by_card.items():
            played = sum(int(r["played"] or 0) for r in card_rows)
            if played < min_played:
                continue
            drawn = sum(int(r["drawn"] or 0) for r in card_rows)
            wins_wp = sum(int(r["wins_when_played"] or 0) for r in card_rows)
            games_wp = sum(int(r["games_when_played"] or 0) for r in card_rows)

            if games_wp > 0:
                agg_win_rate: float | None = wins_wp / games_wp
            else:
                wr_vals = [float(r["win_rate"]) for r in card_rows if r["win_rate"] is not None]
                agg_win_rate = (sum(wr_vals) / len(wr_vals)) if wr_vals else None

            if drawn > 0:
                agg_play_rate: float | None = played / drawn
            else:
                pr_vals = [float(r["play_rate"]) for r in card_rows if r["play_rate"] is not None]
                agg_play_rate = (sum(pr_vals) / len(pr_vals)) if pr_vals else None

            play_share = (played / total_played_all) if total_played_all > 0 else 0.0
            latest = card_rows[-1]
            latest_wr = float(latest["win_rate"]) if latest["win_rate"] is not None else agg_win_rate
            prev_wr = (
                float(card_rows[-2]["win_rate"])
                if len(card_rows) >= 2 and card_rows[-2]["win_rate"] is not None
                else None
            )
            delta_wr = (latest_wr - prev_wr) if (latest_wr is not None and prev_wr is not None) else 0.0

            out.append(
                sanitize_json_value(
                    {
                        "card_name": card_name,
                        "samples": len(card_rows),
                        "latest_step": int(latest["step"]),
                        "played": played,
                        "drawn": drawn,
                        "wins_when_played": wins_wp,
                        "games_when_played": games_wp,
                        "play_share": play_share,
                        "play_rate": agg_play_rate,
                        "win_rate": agg_win_rate,
                        "latest_play_rate": float(latest["play_rate"]) if latest["play_rate"] is not None else agg_play_rate,
                        "latest_win_rate": latest_wr,
                        "delta_win_rate": delta_wr,
                    }
                )
            )

        out.sort(key=lambda item: (-int(item["played"]), item["card_name"]))
        return out[:limit]

    def get_card_trends(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        cards: list[str] | None = None,
        limit_cards: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        selected = [c for c in (cards or []) if c]
        if not selected:
            meta = self.get_card_meta(project=project, run_id=run_id, limit=limit_cards)
            selected = [m["card_name"] for m in meta]
        if not selected:
            return {}

        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("cs.run_id = ?")
            params.append(run_id)
        placeholders = ", ".join("?" for _ in selected)
        clauses.append(f"cs.card_name IN ({placeholders})")
        params.extend(selected)
        where = f"WHERE {' AND '.join(clauses)}"

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT cs.card_name,
                       cs.step,
                       cs.run_id,
                       cs.played,
                       cs.drawn,
                       cs.wins_when_played,
                       cs.games_when_played,
                       cs.play_rate,
                       cs.win_rate,
                       cs.recorded_at
                FROM card_stats cs
                JOIN projects p ON p.id = cs.project_id
                {where}
                ORDER BY cs.step ASC, cs.id ASC
                """,
                tuple(params),
            ).fetchall()

        series_map: dict[str, list[dict[str, Any]]] = {c: [] for c in selected}
        for r in rows:
            item = dict(r)
            card = str(item.pop("card_name"))
            series_map.setdefault(card, []).append(sanitize_json_value(item))
        return series_map

    def export_card_stats(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("(p.name = ? OR p.id = ?)")
            params.extend([project, project])
        if run_id:
            clauses.append("cs.run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT p.name AS project_name,
                       cs.run_id,
                       cs.step,
                       cs.card_name,
                       cs.played,
                       cs.drawn,
                       cs.wins_when_played,
                       cs.games_when_played,
                       cs.play_rate,
                       cs.win_rate,
                       cs.extra_json,
                       cs.recorded_at
                FROM card_stats cs
                JOIN projects p ON p.id = cs.project_id
                {where}
                ORDER BY cs.step ASC, cs.card_name ASC, cs.id ASC
                """,
                tuple(params),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["extra"] = json.loads(d.pop("extra_json", "{}") or "{}")
                out.append(sanitize_json_value(d))
            return out

    def insert_distribution(
        self,
        *,
        run_id: str,
        name: str,
        step: int,
        labels: list[str],
        probs: list[float],
        entropy: float | None = None,
        recorded_at: int | None = None,
    ) -> dict[str, Any]:
        import math

        ts = int(recorded_at or now_ms())
        clean_labels = [str(lb) for lb in labels]
        clean_probs: list[float] = []
        for x in probs:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                fx = 0.0
            clean_probs.append(fx if math.isfinite(fx) else 0.0)

        ent = self._clean_opt_float(entropy)
        if ent is None and clean_probs:
            total = sum(max(0.0, x) for x in clean_probs)
            if total > 0:
                ent = -sum(
                    (max(0.0, x) / total) * math.log(max(0.0, x) / total)
                    for x in clean_probs
                    if x > 0
                )
        ent = self._clean_opt_float(ent)

        with self._lock:
            row = self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO distributions (run_id, step, name, labels_json, probs_json, entropy, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(step),
                        str(name),
                        _json_dumps(clean_labels),
                        _json_dumps(clean_probs),
                        ent,
                        ts,
                    ),
                )
        return sanitize_json_value(
            {
                "run_id": run_id,
                "name": str(name),
                "step": int(step),
                "labels": clean_labels,
                "probs": clean_probs,
                "entropy": ent,
                "recorded_at": ts,
            }
        )

    def list_distribution_names(self, run_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT name FROM distributions WHERE run_id = ? ORDER BY name ASC",
                (run_id,),
            ).fetchall()
            return [str(r["name"]) for r in rows]

    def get_distribution_series(
        self,
        run_id: str,
        name: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if name:
            clauses.append("name = ?")
            params.append(name)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT step, name, labels_json, probs_json, entropy, recorded_at
                FROM distributions
                WHERE {where}
                ORDER BY step ASC, id ASC
                """,
                tuple(params),
            ).fetchall()

        points = [
            sanitize_json_value(
                {
                    "step": int(r["step"]),
                    "name": str(r["name"]),
                    "labels": json.loads(r["labels_json"] or "[]"),
                    "probs": json.loads(r["probs_json"] or "[]"),
                    "entropy": r["entropy"],
                    "recorded_at": int(r["recorded_at"]),
                }
            )
            for r in rows
        ]
        if len(points) > limit and limit >= 2:
            # Uniform index downsampling preserving first & last step.
            idxs = [round(i * (len(points) - 1) / (limit - 1)) for i in range(limit)]
            dedup_idxs = list(dict.fromkeys(idxs))
            points = [points[i] for i in dedup_idxs]
        return points

"""Tests for League series, N-player matches, H2H matrix, checkpoint lineage/upload, and league.db import."""

from __future__ import annotations

import math
import socket
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import uvicorn

import traininglog as tl
from traininglog.server.app import create_app
from traininglog.server.config import ServerConfig
from traininglog.server.ratings import Glicko2Rating, rate_glicko2_match


def _start_server(
    data_dir: Path, alert_transport: httpx.AsyncBaseTransport | None = None
) -> tuple[str, uvicorn.Server, threading.Thread]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    cfg = ServerConfig(
        data_dir=data_dir,
        host="127.0.0.1",
        port=port,
        stall_timeout_sec=300,
        stall_check_interval_sec=60,
    )
    app = create_app(cfg, alert_transport=alert_transport)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server, thread


def test_checkpoints_lineage_upload_and_league_series(tmp_path: Path) -> None:
    alerts_sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        alerts_sent.append({"url": str(request.url), "body": request.content})
        return httpx.Response(200, json={"ok": True})

    url, server, thread = _start_server(
        tmp_path / "data", alert_transport=httpx.MockTransport(handler)
    )
    try:
        with httpx.Client(base_url=url) as client:
            client.post(
                "/api/alerts/rules",
                json={
                    "name": "best-elo-hook",
                    "channel": "webhook",
                    "target": {"url": "http://hook.local/best"},
                    "events": ["new_best_elo"],
                    "cooldown_sec": 0,
                },
            )

        # Hybrid checkpoints: path_ref + uploaded blob + parent lineage
        weights_file = tmp_path / "dual_v2.pt"
        weights_bytes = b"fake-pytorch-weights-payload-12345"
        weights_file.write_bytes(weights_bytes)

        c_bot = tl.log_checkpoint(
            "builtin:Heuristic_Bot", project="cardgame-ppo", name="Heuristic_Bot", server_url=url
        )
        assert c_bot["is_bot"] is True

        c_v1 = tl.log_checkpoint(
            "runs/saved_runs/Dual_v1.pt",
            project="cardgame-ppo",
            name="Dual_v1",
            step=100_000_000,
            arch="dual_stream",
            manifest_hash="abc123",
            tags=["saved_run"],
            server_url=url,
        )
        c_v2 = tl.log_checkpoint(
            weights_file,
            project="cardgame-ppo",
            name="Dual_v2",
            step=200_000_000,
            parent="Dual_v1",
            arch="dual_stream",
            manifest_hash="abc123",
            tags=["best"],
            pinned=True,
            upload=True,
            server_url=url,
        )
        assert c_v2["parent_id"] == c_v1["id"]
        assert c_v2["storage_mode"] == "uploaded"
        assert c_v2["size_bytes"] == len(weights_bytes)

        # 1000-game series with exact P1/P2 splits & decks
        res = tl.log_match_series(
            project="cardgame-ppo",
            model_a="Dual_v2",
            model_b="Heuristic_Bot",
            total_games=1000,
            a_wins=780,
            b_wins=220,
            a_as_p1_wins=405,
            a_as_p1_games=500,
            a_as_p2_wins=375,
            a_as_p2_games=500,
            deck_a="Starter_Dwarven",
            deck_b="Starter_Heaven",
            avg_rounds=6.2,
            avg_steps=34.5,
            server_url=url,
        )
        assert res["duplicate"] is False
        assert len(alerts_sent) == 1  # new_best_elo alert fired!

        # 4-player OpenSkill match
        os_res = tl.log_match(
            project="cardgame-ppo",
            mode="multiplayer",
            turns=9,
            participants=[
                {"agent": "Dual_v2", "seat": 0, "rank": 1, "score": 20},
                {"agent": "Dual_v1", "seat": 1, "rank": 2, "score": 10},
                {"agent": "Heuristic_Bot", "seat": 2, "rank": 3, "score": 4},
                {"agent": "Random_Bot", "seat": 3, "rank": 4, "score": 0},
            ],
            server_url=url,
        )
        assert os_res["duplicate"] is False

        # Direct pre-computed rating logging
        tl.log_rating(
            project="cardgame-ppo",
            agent="Scripted_Baseline",
            elo=1620.0,
            glicko_mu=1620.0,
            glicko_rd=60.0,
            server_url=url,
        )

        with httpx.Client(base_url=url) as client:
            # Download uploaded blob matches byte-for-byte
            dl = client.get(f"/api/checkpoints/{c_v2['id']}/download")
            assert dl.status_code == 200
            assert dl.content == weights_bytes

            # Lineage DAG
            lin = client.get("/api/checkpoints/lineage?project=cardgame-ppo").json()
            assert any(e["from"] == c_v1["id"] and e["to"] == c_v2["id"] for e in lin["edges"])

            # Leaderboard filtering (exclude bots, filter arch)
            lb_nobots = client.get(
                "/api/league/leaderboard?project=cardgame-ppo&include_bots=false&arch=dual_stream"
            ).json()["entries"]
            assert all(not e["is_bot"] for e in lb_nobots)
            assert lb_nobots[0]["name"] == "Dual_v2"
            assert lb_nobots[0]["glicko_mu"] > 1500.0

            # H2H matrix
            mat = client.get("/api/league/matrix?project=cardgame-ppo").json()
            cell = mat["cells"][c_v2["id"]][c_bot["id"]]
            assert cell["total_games"] == 1000
            assert math.isclose(cell["win_rate"], 0.78, abs_tol=1e-6)
            assert math.isclose(cell["as_p1_wr"], 405 / 500, abs_tol=1e-6)
            assert math.isclose(cell["as_p2_wr"], 375 / 500, abs_tol=1e-6)

            # Seat & deck breakdown
            seats = client.get("/api/league/seats?project=cardgame-ppo").json()
            assert seats["side_split"]["total_games"] >= 1000
            assert len(seats["deck_matchups"]) == 1
            assert seats["deck_matchups"][0]["deck_a"] == "Starter_Dwarven"

            # Recompute produces identical rating
            mu_before = lb_nobots[0]["glicko_mu"]
            recomp = client.post("/api/league/recompute?project=cardgame-ppo").json()
            assert recomp["recomputed_matches"] == 2
            lb_after = client.get(
                "/api/league/leaderboard?project=cardgame-ppo&include_bots=false&arch=dual_stream"
            ).json()["entries"]
            assert math.isclose(lb_after[0]["glicko_mu"], mu_before, abs_tol=1e-6)
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_import_league_db_exact_parity_and_idempotency(tmp_path: Path) -> None:
    # Build a synthetic cardgame league.db matching agents/league.py schema
    league_db_path = tmp_path / "league.db"
    conn = sqlite3.connect(str(league_db_path))
    conn.executescript(
        """
        CREATE TABLE models (
            name TEXT PRIMARY KEY,
            checkpoint_path TEXT NOT NULL,
            arch TEXT DEFAULT 'unknown',
            rating REAL DEFAULT 1500.0,
            rd REAL DEFAULT 350.0,
            vol REAL DEFAULT 0.06,
            matches_played INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            p1_wins INTEGER DEFAULT 0,
            p1_games INTEGER DEFAULT 0,
            p2_wins INTEGER DEFAULT 0,
            p2_games INTEGER DEFAULT 0,
            total_rounds REAL DEFAULT 0.0,
            total_steps REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_played TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            total_games INTEGER NOT NULL,
            a_wins INTEGER NOT NULL,
            b_wins INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            a_as_p1_wins INTEGER NOT NULL,
            a_as_p1_games INTEGER NOT NULL,
            a_as_p2_wins INTEGER NOT NULL,
            a_as_p2_games INTEGER NOT NULL,
            b_as_p1_wins INTEGER NOT NULL,
            b_as_p1_games INTEGER NOT NULL,
            b_as_p2_wins INTEGER NOT NULL,
            b_as_p2_games INTEGER NOT NULL,
            p1_side_wins INTEGER NOT NULL,
            p2_side_wins INTEGER NOT NULL,
            p1_side_winrate REAL NOT NULL,
            p2_side_winrate REAL NOT NULL,
            avg_rounds REAL NOT NULL,
            avg_steps REAL NOT NULL,
            a_rating_before REAL NOT NULL,
            a_rd_before REAL NOT NULL,
            a_rating_after REAL NOT NULL,
            a_rd_after REAL NOT NULL,
            b_rating_before REAL NOT NULL,
            b_rd_before REAL NOT NULL,
            b_rating_after REAL NOT NULL,
            b_rd_after REAL NOT NULL
        );
        """
    )

    r_arc = Glicko2Rating()
    r_bot = Glicko2Rating()
    r_arc1, r_bot1 = rate_glicko2_match(r_arc, r_bot, 0.84)

    r_tr = Glicko2Rating()
    r_arc2, r_tr1 = rate_glicko2_match(r_arc1, r_tr, 0.58)

    with conn:
        conn.execute(
            "INSERT INTO models (name, checkpoint_path, arch, rating, rd, vol, games_played, wins, losses) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Random_Bot", "builtin:random", "bot", r_bot1.mu, r_bot1.rd, r_bot1.vol, 1000, 160, 840),
        )
        conn.execute(
            "INSERT INTO models (name, checkpoint_path, arch, rating, rd, vol, games_played, wins, losses) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ArcWarden", "/abs/runs/saved_runs/ArcWarden.pt", "dual_stream", r_arc2.mu, r_arc2.rd, r_arc2.vol, 2000, 1420, 580),
        )
        conn.execute(
            "INSERT INTO models (name, checkpoint_path, arch, rating, rd, vol, games_played, wins, losses) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Lion", "/abs/runs/saved_runs/Lion.pt", "transformer", r_tr1.mu, r_tr1.rd, r_tr1.vol, 1000, 420, 580),
        )
        conn.execute(
            """
            INSERT INTO matches (
                timestamp, model_a, model_b, total_games, a_wins, b_wins, draws,
                a_as_p1_wins, a_as_p1_games, a_as_p2_wins, a_as_p2_games,
                b_as_p1_wins, b_as_p1_games, b_as_p2_wins, b_as_p2_games,
                p1_side_wins, p2_side_wins, p1_side_winrate, p2_side_winrate,
                avg_rounds, avg_steps,
                a_rating_before, a_rd_before, a_rating_after, a_rd_after,
                b_rating_before, b_rd_before, b_rating_after, b_rd_after
            ) VALUES ('2026-11-20 12:00:00', 'ArcWarden', 'Random_Bot', 1000, 840, 160, 0,
                      430, 500, 410, 500, 90, 500, 70, 500, 520, 480, 52.0, 48.0, 6.1, 35.0,
                      ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (r_arc.mu, r_arc.rd, r_arc1.mu, r_arc1.rd, r_bot.mu, r_bot.rd, r_bot1.mu, r_bot1.rd),
        )
        conn.execute(
            """
            INSERT INTO matches (
                timestamp, model_a, model_b, total_games, a_wins, b_wins, draws,
                a_as_p1_wins, a_as_p1_games, a_as_p2_wins, a_as_p2_games,
                b_as_p1_wins, b_as_p1_games, b_as_p2_wins, b_as_p2_games,
                p1_side_wins, p2_side_wins, p1_side_winrate, p2_side_winrate,
                avg_rounds, avg_steps,
                a_rating_before, a_rd_before, a_rating_after, a_rd_after,
                b_rating_before, b_rd_before, b_rating_after, b_rd_after
            ) VALUES ('2026-11-20 12:05:00', 'ArcWarden', 'Lion', 1000, 580, 420, 0,
                      300, 500, 280, 500, 220, 500, 200, 500, 520, 480, 52.0, 48.0, 7.3, 42.0,
                      ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (r_arc1.mu, r_arc1.rd, r_arc2.mu, r_arc2.rd, r_tr.mu, r_tr.rd, r_tr1.mu, r_tr1.rd),
        )
    conn.close()

    url, server, thread = _start_server(tmp_path / "data")
    try:
        res1 = tl.import_league_db(league_db_path, project="imported-league", server_url=url)
        assert res1["models_imported"] == 3
        assert res1["matches_imported"] == 2
        assert res1["matches_skipped"] == 0

        # Second import is completely idempotent
        res2 = tl.import_league_db(league_db_path, project="imported-league", server_url=url)
        assert res2["matches_imported"] == 0
        assert res2["matches_skipped"] == 2

        with httpx.Client(base_url=url) as client:
            lb = client.get("/api/league/leaderboard?project=imported-league").json()["entries"]
            by_name = {e["name"]: e for e in lb}
            assert math.isclose(by_name["ArcWarden"]["glicko_mu"], r_arc2.mu, abs_tol=1e-3)
            assert math.isclose(by_name["Lion"]["glicko_mu"], r_tr1.mu, abs_tol=1e-3)
            assert math.isclose(by_name["Random_Bot"]["glicko_mu"], r_bot1.mu, abs_tol=1e-3)
            assert by_name["ArcWarden"]["arch"] == "dual_stream"
            assert "saved_run" in by_name["ArcWarden"]["tags"]
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

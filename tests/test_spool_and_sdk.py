"""Tests for disk WAL spool outage resilience, idempotent sync, SummaryWriter, wandb_compat, and console capture."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

import traininglog as tl
import traininglog.wandb_compat as wandb
from traininglog.sdk.spool import iter_unsynced_run_ids
from traininglog.server.app import create_app
from traininglog.server.config import ServerConfig


def _alloc_port() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock, int(sock.getsockname()[1])


def _start_server(data_dir: Path, sock: socket.socket) -> tuple[uvicorn.Server, threading.Thread]:
    cfg = ServerConfig(
        data_dir=data_dir,
        host="127.0.0.1",
        port=int(sock.getsockname()[1]),
        stall_timeout_sec=300,
        stall_check_interval_sec=60,
    )
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cfg.port, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    return server, thread


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5.0)


def test_offline_spool_sync_and_idempotent_dedup(tmp_path: Path) -> None:
    sock, port = _alloc_port()
    dead_url = f"http://127.0.0.1:{port}"
    spool_dir = tmp_path / "spool"

    # Log while server port is NOT yet listening
    sock.close()
    run = tl.init(
        project="cardgame-ppo",
        name="outage-run",
        run_id="run_outage_1",
        config={"lr": 2.5e-4, "arch": "dual_stream"},
        server_url=dead_url,
        spool_dir=spool_dir,
        capture_console=False,
        capture_system=False,
        finish_timeout=0.2,
    )
    for i in range(1, 6):
        run.log({"losses/policy_loss": 0.5 / i, "charts/SPS": 200_000 + i}, step=i * 1000)
    run.finish(exit_code=0, timeout=0.2)

    assert "run_outage_1" in iter_unsynced_run_ids(spool_dir)

    # Now start the server on a fresh socket/port and sync
    sock2, port2 = _alloc_port()
    url2 = f"http://127.0.0.1:{port2}"
    server, thread = _start_server(tmp_path / "data", sock2)
    try:
        res1 = tl.sync(spool_dir=spool_dir, server_url=url2)
        assert res1["errors"] == []
        assert "run_outage_1" in res1["synced_runs"]
        assert res1["synced_events"] >= 5
        assert iter_unsynced_run_ids(spool_dir) == []

        # Re-send partial/overlapping event batch directly to verify idempotency
        with httpx.Client(base_url=url2) as client:
            detail = client.get("/api/runs/run_outage_1").json()
            assert detail["run"]["state"] == "finished"
            series = client.get(
                "/api/runs/run_outage_1/series?metrics=losses/policy_loss"
            ).json()
            assert series["series"]["losses/policy_loss"]["total_points"] == 5

            # Replay overlapping event_seq=1..5 + new event_seq=999
            dup_resp = client.post(
                "/api/runs/run_outage_1/events",
                json={
                    "seq": 999,
                    "events": [
                        {
                            "event_seq": 1,
                            "type": "scalars",
                            "step": 1000,
                            "wall_ms": 123456,
                            "metrics": {"losses/policy_loss": 99.0},
                        },
                        {
                            "event_seq": 999,
                            "type": "scalars",
                            "step": 6000,
                            "wall_ms": 123457,
                            "metrics": {"losses/policy_loss": 0.05},
                        },
                    ],
                },
            ).json()
            assert dup_resp["accepted"] == 1

            series_after = client.get(
                "/api/runs/run_outage_1/series?metrics=losses/policy_loss"
            ).json()
            assert series_after["series"]["losses/policy_loss"]["total_points"] == 6
            assert series_after["series"]["losses/policy_loss"]["y"][0] == 0.5
    finally:
        _stop_server(server, thread)


def test_live_server_outage_recovery_while_running(tmp_path: Path) -> None:
    # Bind port first so nobody steals it, close momentarily? Or use a known free port:
    sock, port = _alloc_port()
    sock.close()
    url = f"http://127.0.0.1:{port}"
    spool_dir = tmp_path / "spool"

    run = tl.init(
        project="cardgame-ppo",
        name="live-recovery",
        run_id="run_recover_1",
        server_url=url,
        spool_dir=spool_dir,
        capture_console=False,
        capture_system=False,
    )
    run.log({"game/win_rate_overall": 0.55}, step=10_000)
    time.sleep(0.3)

    # Start server AFTER logging began
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock2.bind(("127.0.0.1", port))
    server, thread = _start_server(tmp_path / "data", sock2)
    try:
        assert run.flush(timeout=5.0) is True
        run.log({"game/win_rate_overall": 0.68}, step=20_000)
        run.finish(timeout=5.0)

        with httpx.Client(base_url=url) as client:
            series = client.get(
                "/api/runs/run_recover_1/series?metrics=game/win_rate_overall"
            ).json()
            assert series["series"]["game/win_rate_overall"]["y"] == [0.55, 0.68]
    finally:
        _stop_server(server, thread)


def test_summary_writer_wandb_compat_and_crash_console(tmp_path: Path) -> None:
    sock, port = _alloc_port()
    url = f"http://127.0.0.1:{port}"
    server, thread = _start_server(tmp_path / "data", sock)
    spool_dir = tmp_path / "spool"

    try:
        # 1. SummaryWriter + crash traceback capture
        with pytest.raises(RuntimeError, match="boom"):
            with tl.init(
                project="cardgame-ppo",
                name="crash-run",
                run_id="run_crash_1",
                server_url=url,
                spool_dir=spool_dir,
                capture_console=True,
                capture_system=False,
            ) as run:
                writer = run.summary_writer()
                print("Starting PPO rollout update 1")
                writer.add_scalar("losses/policy_loss", 0.25, 1000)
                writer.add_scalars("rewards", {"damage": 1.2, "kills": 0.5}, 1000)
                writer.add_histogram("actions/lane_placement_histogram", np.array([0, 1, 1, 2, 3]), 1000, bins=4)
                writer.add_text("hyperparameters", "|lr|1e-4|", 0)
                raise RuntimeError("boom during update")

        with httpx.Client(base_url=url) as client:
            detail = client.get("/api/runs/run_crash_1").json()
            assert detail["run"]["state"] == "crashed"
            assert any(a["kind"] == "crashed" for a in detail["open_anomalies"])

            console = client.get("/api/runs/run_crash_1/console").json()
            joined = "\n".join(line["text"] for line in console["lines"])
            assert "Starting PPO rollout update 1" in joined
            assert "RuntimeError: boom during update" in joined

            hists = client.get(
                "/api/runs/run_crash_1/histograms?name=actions/lane_placement_histogram"
            ).json()
            assert len(hists["snapshots"]) == 1
            assert hists["snapshots"][0]["total_count"] == 5

        # 2. wandb_compat shim
        wrun = wandb.init(
            project="cleanRL",
            name="wandb-shim-run",
            id="run_wandb_1",
            sync_tensorboard=True,
            config={"learning_rate": 1.5e-4},
            server_url=url,
            spool_dir=spool_dir,
            capture_console=False,
            capture_system=False,
        )
        tbl = wandb.Table(
            data=[["Mica", 120, 42.5]],
            columns=["Card Name", "Play Count", "Play Rate (%)"],
        )
        wandb.log(
            {
                "cards/play_distribution": wandb.plot.bar(tbl, "Card Name", "Play Rate (%)", title="Most Played"),
                "game/win_rate_overall": 0.72,
                "global_step": 50_000,
            }
        )
        wandb.finish()

        with httpx.Client(base_url=url) as client:
            wdetail = client.get("/api/runs/run_wandb_1").json()
            assert wdetail["run"]["state"] == "finished"
            assert wdetail["run"]["summary"]["game/win_rate_overall"] == 0.72
            assert "cards/play_distribution" in wdetail["run"]["summary"]
    finally:
        _stop_server(server, thread)

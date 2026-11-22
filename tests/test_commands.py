"""Tests for cooperative remote run commands (checkpoint, eval, stop)."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

import traininglog as tl
from traininglog.server.app import create_app
from traininglog.server.config import ServerConfig


def _start_server(data_dir: Path) -> tuple[str, uvicorn.Server, threading.Thread]:
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
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server, thread


def test_cooperative_command_roundtrip(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        run = tl.init(
            project="cardgame-ppo",
            name="cmd-test",
            run_id="run_cmd_1",
            server_url=url,
            spool_dir=tmp_path / "spool",
            capture_console=False,
            capture_system=False,
            command_poll_interval=0.1,
        )
        run.log({"charts/SPS": 300_000}, step=1000)
        assert run.flush(timeout=5.0) is True
        assert run.should_stop is False

        with httpx.Client(base_url=url) as client:
            c1 = client.post(
                "/api/runs/run_cmd_1/commands",
                json={"cmd_type": "checkpoint", "payload": {}},
            )
            assert c1.status_code == 200
            c2 = client.post(
                "/api/runs/run_cmd_1/commands",
                json={"cmd_type": "stop", "payload": {}},
            )
            assert c2.status_code == 200

        # Wait briefly for background poller to pick up commands
        received = []
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and len(received) < 2:
            received.extend(run.poll_commands())
            if len(received) < 2:
                time.sleep(0.05)

        types = {c.type for c in received}
        assert types == {"checkpoint", "stop"}
        assert run.should_stop is True

        for c in received:
            if c.type == "checkpoint":
                c.complete("Saved ppo_step_1000.pt")
            else:
                c.ack("Stopping gracefully")

        run.finish(timeout=5.0)

        with httpx.Client(base_url=url) as client:
            cmds = client.get("/api/runs/run_cmd_1/commands").json()["commands"]
            by_type = {c["cmd_type"]: c for c in cmds}
            assert by_type["checkpoint"]["status"] == "completed"
            assert by_type["checkpoint"]["result_message"] == "Saved ppo_step_1000.pt"
            assert by_type["stop"]["status"] == "acked"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

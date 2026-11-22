"""Tests for ReDo dormant neuron & effective rank (srank) plasticity reports and alerts."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn

import traininglog as tl
from traininglog.server.app import create_app
from traininglog.server.config import ServerConfig


@dataclass
class _MockLayerResult:
    name: str
    hook_type: str
    dim: int
    hard_dead_count: int
    hard_dead_pct: float
    dormant_count: int
    dormant_pct: float
    weak_count: int
    weak_pct: float
    effective_rank: float
    rank_ratio: float


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


def test_plasticity_json_and_dataclass_ingestion_and_alerts(tmp_path: Path) -> None:
    alerts_sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        alerts_sent.append(request.content)
        return httpx.Response(200, json={"ok": True})

    url, server, thread = _start_server(
        tmp_path / "data", alert_transport=httpx.MockTransport(handler)
    )
    try:
        with httpx.Client(base_url=url) as client:
            client.post(
                "/api/alerts/rules",
                json={
                    "name": "plasticity-hook",
                    "channel": "webhook",
                    "target": {"url": "http://hook.local/plast"},
                    "events": ["high_dormancy", "rank_collapse"],
                    "cooldown_sec": 0,
                },
            )

        # 1. check_dormant_neurons.py --json file format (rank_ratio_pct)
        json_file = tmp_path / "redo_report.json"
        json_file.write_text(
            json.dumps(
                {
                    "meta": {
                        "model_name": "Axe5000M",
                        "arch": "dual_stream",
                        "total_obs": 4096,
                        "hook_mode": "post_activation",
                    },
                    "layers": [
                        {
                            "name": "encoder.0",
                            "hook_type": "post_GELU",
                            "dim": 256,
                            "hard_dead_count": 2,
                            "hard_dead_pct": 0.78,
                            "dormant_count": 12,
                            "dormant_pct": 4.69,
                            "weak_count": 30,
                            "weak_pct": 11.72,
                            "effective_rank": 140.0,
                            "rank_ratio_pct": 54.69,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        res_healthy = tl.log_plasticity(
            project="cardgame-ppo", json_path=json_file, step=500_000_000, server_url=url
        )
        assert res_healthy["anomalies"] == []
        assert math.isclose(res_healthy["report"]["dormant_pct"], 4.6875, abs_tol=1e-2)

        # 2. Run-linked dataclass report triggering critical dormancy + rank collapse
        run = tl.init(
            project="cardgame-ppo",
            name="collapsed-run",
            run_id="run_plast_1",
            arch="transformer",
            server_url=url,
            spool_dir=tmp_path / "spool",
            capture_console=False,
            capture_system=False,
        )
        res_bad = run.log_plasticity(
            step=1_000_000_000,
            layers=[
                _MockLayerResult(
                    name="attn.ffn.0",
                    hook_type="post_ReLU",
                    dim=256,
                    hard_dead_count=60,
                    hard_dead_pct=23.4,
                    dormant_count=110,
                    dormant_pct=43.0,
                    weak_count=160,
                    weak_pct=62.5,
                    effective_rank=32.0,
                    rank_ratio=12.5,
                )
            ],
        )
        run.finish(timeout=5.0)

        kinds = {a["kind"] for a in res_bad["anomalies"]}
        assert kinds == {"high_dormancy", "rank_collapse"}
        assert len(alerts_sent) == 2

        with httpx.Client(base_url=url) as client:
            listed = client.get("/api/plasticity?run_id=run_plast_1").json()["reports"]
            assert len(listed) == 1
            assert listed[0]["layers"][0]["rank_ratio_pct"] == 12.5
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

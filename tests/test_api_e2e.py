"""End-to-end API tests: multi-run comparison, config diff, auth token, export, and PWA assets."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

import traininglog as tl
from traininglog.server.app import create_app
from traininglog.server.config import ServerConfig


def _start_server(
    data_dir: Path, api_token: str | None = None
) -> tuple[str, uvicorn.Server, threading.Thread]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    cfg = ServerConfig(
        data_dir=data_dir,
        host="127.0.0.1",
        port=port,
        api_token=api_token,
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


def test_compare_config_diff_export_and_static_pwa(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    spool = tmp_path / "spool"
    try:
        r1 = tl.init(
            project="cardgame-ppo",
            name="dual-run",
            run_id="run_cmp_1",
            arch="dual_stream",
            config={"lr": 1.5e-4, "gamma": 0.99, "arch": "dual_stream"},
            server_url=url,
            spool_dir=spool,
            capture_console=False,
            capture_system=False,
        )
        r2 = tl.init(
            project="cardgame-ppo",
            name="mlp-run",
            run_id="run_cmp_2",
            arch="mlp",
            config={"lr": 2.5e-4, "gamma": 0.99, "arch": "mlp"},
            server_url=url,
            spool_dir=spool,
            capture_console=False,
            capture_system=False,
        )
        r1.log({"game/win_rate_overall": 0.62, "losses/policy_loss": 0.14}, step=100_000_000)
        r2.log({"game/win_rate_overall": 0.48, "losses/policy_loss": 0.21}, step=100_000_000)
        r1.finish(timeout=5.0)
        r2.finish(timeout=5.0)

        with httpx.Client(base_url=url) as client:
            # Multi-run series compare
            cmp_res = client.post(
                "/api/compare/series",
                json={"run_ids": ["run_cmp_1", "run_cmp_2"], "metrics": ["game/win_rate_overall"]},
            )
            assert cmp_res.status_code == 200
            cmp_json = cmp_res.json()
            assert cmp_json["series"]["game/win_rate_overall"]["run_cmp_1"]["y"] == [0.62]
            assert cmp_json["series"]["game/win_rate_overall"]["run_cmp_2"]["y"] == [0.48]

            # Config diff
            diff_res = client.post(
                "/api/compare/config-diff",
                json={"run_ids": ["run_cmp_1", "run_cmp_2"]},
            )
            assert diff_res.status_code == 200
            rows = {row["key"]: row for row in diff_res.json()["rows"]}
            assert rows["lr"]["differs"] is True
            assert rows["arch"]["differs"] is True
            assert rows["gamma"]["differs"] is False

            # CSV & JSON exports
            csv_resp = client.get("/api/runs/run_cmp_1/export?format=csv")
            assert csv_resp.status_code == 200
            assert "100000000" in csv_resp.text
            assert "game/win_rate_overall" in csv_resp.text

            json_resp = client.get("/api/runs/run_cmp_1/export?format=json")
            assert json_resp.status_code == 200
            exported = json.loads(json_resp.text)
            assert exported["run"]["id"] == "run_cmp_1"
            assert len(exported["scalars"]) == 2

            # Static PWA & vendored assets served cleanly
            for asset in [
                "/",
                "/manifest.webmanifest",
                "/sw.js",
                "/icon.svg",
                "/static/css/app.css",
                "/static/js/main.js",
                "/static/js/api.js",
                "/static/js/presets.js",
                "/static/js/uplot_chart.js",
                "/static/js/views/active.js",
                "/static/js/views/runs.js",
                "/static/js/views/run_detail.js",
                "/static/js/views/compare.js",
                "/static/js/views/league.js",
                "/static/js/views/replays.js",
                "/static/js/views/cards.js",
                "/static/js/views/alerts.js",
                "/static/js/components/plasticity_panel.js",
                "/static/vendor/uPlot.iife.min.js",
                "/static/vendor/uPlot.min.css",
                "/static/vendor/preact.module.js",
                "/static/vendor/hooks.module.js",
                "/static/vendor/htm.module.js",
                "/static/vendor/ui.mjs",
            ]:
                r = client.get(asset)
                assert r.status_code == 200, f"Asset {asset} failed with {r.status_code}"
                assert len(r.content) > 0

            # Warm non-blue palette check
            css_text = client.get("/static/css/app.css").text
            assert "#e59838" in css_text and "#0c0c0b" in css_text
            for blue_hex in ("#38bdf8", "#0b0f17", "#121826", "#0284c7", "#08121e", "#06090f"):
                assert blue_hex not in css_text.lower()
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_optional_api_token_enforcement(tmp_path: Path) -> None:
    secret = "super-secret-vpn-token"
    url, server, thread = _start_server(tmp_path / "data", api_token=secret)
    try:
        with httpx.Client(base_url=url) as client:
            health = client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["auth_required"] is True

            assert client.get("/api/runs").status_code == 401
            assert client.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert client.get("/api/runs", headers={"Authorization": f"Bearer {secret}"}).status_code == 200
            assert client.get(f"/api/runs?token={secret}").status_code == 200
            assert client.get("/").status_code == 200

        # SDK with matching token succeeds
        run = tl.init(
            project="auth-test",
            run_id="run_auth_1",
            server_url=url,
            api_token=secret,
            spool_dir=tmp_path / "spool",
            capture_console=False,
            capture_system=False,
        )
        run.log({"loss": 0.1}, step=1)
        run.finish(timeout=5.0)

        with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {secret}"}) as client:
            assert client.get("/api/runs/run_auth_1").status_code == 200
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

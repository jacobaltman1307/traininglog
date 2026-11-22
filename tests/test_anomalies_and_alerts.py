"""Tests for RL anomaly detection, stall detection/recovery, and all 5 webhook alert channels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from traininglog.server.alerts import AlertDispatcher
from traininglog.server.anomalies import AnomalyDetector
from traininglog.server.db import Database, now_ms
from traininglog.server.sse import EventBus


@pytest.mark.asyncio
async def test_anomaly_detection_rules_and_stall_recovery(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    bus = EventBus()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    dispatcher = AlertDispatcher(db, transport=httpx.MockTransport(handler))
    detector = AnomalyDetector(db, bus, dispatcher)

    run, _ = db.create_or_resume_run(
        {"run_id": "run_anom_1", "project": "cardgame-ppo", "name": "anom-test"}
    )

    # Create a webhook rule for all anomaly types with cooldown 0 so each fires
    db.create_alert_rule(
        {
            "name": "All anomalies",
            "channel": "webhook",
            "target": {"url": "http://example.local/hook"},
            "events": ["nan_inf", "kl_spike", "entropy_collapse", "sps_drop", "stalled"],
            "cooldown_sec": 0,
            "enabled": True,
        }
    )

    # 1. Warm up counts + SPS baseline (10 samples around 200_000)
    warmup = {
        "losses/approx_kl": [(i * 100, 1000 + i, 0.01) for i in range(5)],
        "losses/entropy_hand": [(i * 100, 1000 + i, 1.1) for i in range(10)],
        "losses/entropy_board": [(i * 100, 1000 + i, 0.8) for i in range(10)],
        "charts/SPS": [(i * 100, 1000 + i, 200_000.0) for i in range(10)],
    }
    anoms_warm = await detector.inspect_scalars(run, warmup)
    assert anoms_warm == []

    # 2. Trigger NaN, KL spike, hand entropy collapse, board entropy collapse, SPS drop
    bad = {
        "losses/value_loss": [(1100, 2000, float("nan"))],
        "losses/approx_kl": [(1100, 2000, 0.25)],
        "losses/entropy_hand": [(1100, 2000, 0.01)],
        "losses/entropy_board": [(1100, 2000, 0.02)],
        "charts/SPS": [(1100, 2000, 30_000.0)],
    }
    emitted = await detector.inspect_scalars(run, bad)
    kinds = [a["kind"] for a in emitted]
    assert "nan_inf" in kinds
    assert "kl_spike" in kinds
    assert "sps_drop" in kinds
    assert kinds.count("entropy_collapse") == 2

    # 3. Stall detection and automatic recovery when heartbeat resumes
    with db._lock:
        with db._conn:
            db._conn.execute(
                "UPDATE runs SET last_heartbeat_at = ? WHERE id = ?",
                (now_ms() - 400_000, "run_anom_1"),
            )
    stalled = await detector.check_stalled_runs(stall_timeout_sec=300)
    assert len(stalled) == 1
    assert stalled[0]["kind"] == "stalled"
    assert db.get_run("run_anom_1")["state"] == "stalled"

    revived_run, revived = db.apply_run_ingest_state("run_anom_1", heartbeat_ms=now_ms())
    assert revived is True
    assert revived_run is not None
    assert revived_run["state"] == "running"
    open_stalled = [
        a for a in db.list_anomalies(run_id="run_anom_1", resolved=0) if a["kind"] == "stalled"
    ]
    assert open_stalled == []

    assert len(captured) >= 5
    db.close()


@pytest.mark.asyncio
async def test_alert_channels_payloads_and_cooldown(tmp_path: Path) -> None:
    db = Database(tmp_path / "alerts.db")
    requests: list[tuple[str, dict[str, str], bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), dict(request.headers), request.content))
        return httpx.Response(200, json={"status": 1})

    dispatcher = AlertDispatcher(db, transport=httpx.MockTransport(handler))
    run, _ = db.create_or_resume_run(
        {"run_id": "run_alert_1", "project": "cardgame-ppo", "name": "alert-run"}
    )

    rules = [
        db.create_alert_rule(
            {
                "name": "ntfy-phone",
                "channel": "ntfy",
                "target": {"url": "https://ntfy.local/cardgame", "token": "tk_ntfy"},
                "events": ["crashed"],
                "cooldown_sec": 300,
            }
        ),
        db.create_alert_rule(
            {
                "name": "discord-dev",
                "channel": "discord",
                "target": {"url": "https://discord.local/api/webhooks/1/abc"},
                "events": ["crashed"],
                "cooldown_sec": 300,
            }
        ),
        db.create_alert_rule(
            {
                "name": "pushover-dev",
                "channel": "pushover",
                "target": {"url": "https://pushover.local/1/messages.json", "token": "app_tk", "user": "usr_1"},
                "events": ["crashed"],
                "cooldown_sec": 300,
            }
        ),
        db.create_alert_rule(
            {
                "name": "telegram-dev",
                "channel": "telegram",
                "target": {"url": "https://tg.local/bot/sendMessage", "bot_token": "tg_tk", "chat_id": "999"},
                "events": ["crashed"],
                "cooldown_sec": 300,
            }
        ),
        db.create_alert_rule(
            {
                "name": "generic-hook",
                "channel": "webhook",
                "target": {"url": "https://hook.local/events", "headers": {"X-Custom": "yes"}},
                "events": ["crashed"],
                "cooldown_sec": 300,
            }
        ),
    ]

    # First dispatch hits all 5 channels
    res1 = await dispatcher.dispatch_event(
        event_type="crashed",
        project_id=run["project_id"],
        run=run,
        title="[alert-run] Crashed",
        message="OOM in rollout",
        severity="critical",
    )
    assert len(res1) == 5
    assert all(r["status"] == "sent" for r in res1)
    assert len(requests) == 5

    by_url = {u: (hdrs, body) for u, hdrs, body in requests}
    assert by_url["https://ntfy.local/cardgame"][0]["priority"] == "urgent"
    assert by_url["https://ntfy.local/cardgame"][0]["authorization"] == "Bearer tk_ntfy"
    discord_json = json.loads(by_url["https://discord.local/api/webhooks/1/abc"][1])
    assert discord_json["embeds"][0]["title"] == "[alert-run] Crashed"
    assert b"token=app_tk" in by_url["https://pushover.local/1/messages.json"][1]
    tg_json = json.loads(by_url["https://tg.local/bot/sendMessage"][1])
    assert tg_json["chat_id"] == "999"
    hook_hdrs, hook_body = by_url["https://hook.local/events"]
    assert hook_hdrs["x-custom"] == "yes"
    assert json.loads(hook_body)["event_type"] == "crashed"

    # Second dispatch immediately suppressed by 300s cooldown
    res2 = await dispatcher.dispatch_event(
        event_type="crashed",
        project_id=run["project_id"],
        run=run,
        title="[alert-run] Crashed again",
        message="OOM again",
        severity="critical",
    )
    assert res2 == []
    assert len(requests) == 5

    # Test notification helper works directly
    test_ev = await dispatcher.send_test(rules[0]["id"])
    assert test_ev is not None
    assert test_ev["status"] == "sent"
    assert len(requests) == 6
    db.close()

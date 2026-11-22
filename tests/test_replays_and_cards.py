"""Milestone 3 tests: Godot ReplayRecord hub, RL trace blunder detection, MP4 streaming, card meta, and distributions."""

from __future__ import annotations

import csv
import io
import json
import math
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
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


def test_replay_raw_byte_preservation_filtering_and_blob_cleanup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    url, server, thread = _start_server(data_dir)
    try:
        # Deliberately non-canonical whitespace & key order to assert zero re-serialization
        raw_replay_text = (
            '{\n'
            '  "version":  3,\n'
            '  "manifest_hash": "deadbeef12",\n'
            '  "seed1": 1001,\n'
            '  "seed2": 2002,\n'
            '  "deck1": ["Firebolt", "Iron Shieldbearer"],\n'
            '  "deck2": ["Sky Valkyrie", "Light Cleric"],\n'
            '  "config": {"lanes": 4, "starting_hp": 30},\n'
            '  "actions": [\n'
            '    {"type": "play_card", "player": 1, "hand_idx": 0, "board_idx": 2, "category": 0, "index": 0},\n'
            '    {"type": "attack", "player": 2, "hand_idx": -1, "board_idx": 1, "category": 1, "index": 1}\n'
            '  ]\n'
            '}\n'
        )
        raw_bytes = raw_replay_text.encode("utf-8")

        rp1 = tl.log_replay(
            raw_bytes,
            project="cardgame-m3",
            title="Godot Raw Replay #1",
            filename="custom_godot_match.json",
            p1_name="DualStream",
            p2_name="Advanced_Bot",
            winner=1,
            tags=["eval", "raw"],
            server_url=url,
        )
        # Upload second replay sharing exact same raw bytes to verify dedup ref-count deletion
        rp2 = tl.log_replay(
            raw_bytes,
            project="cardgame-m3",
            title="Godot Raw Replay #2",
            filename="custom_godot_match_copy.json",
            server_url=url,
        )

        assert rp1["format"] == "cardgame_replay_json"
        assert rp1["version"] == 3
        assert rp1["manifest_hash"] == "deadbeef12"
        assert rp1["seed1"] == 1001
        assert rp1["seed2"] == 2002
        assert rp1["deck1"] == ["Firebolt", "Iron Shieldbearer"]
        assert rp1["deck2"] == ["Sky Valkyrie", "Light Cleric"]
        assert rp1["action_count"] == 2
        assert rp1["blob_sha256"] == rp2["blob_sha256"]

        with httpx.Client(base_url=url) as client:
            dl = client.get(f"/api/replays/{rp1['id']}/file")
            assert dl.status_code == 200
            assert dl.content == raw_bytes

            # Filtering by deck card, format, has_trace, and query
            by_deck = client.get("/api/replays", params={"project": "cardgame-m3", "deck": "Firebolt"}).json()["replays"]
            assert len(by_deck) == 2

            by_miss = client.get("/api/replays", params={"project": "cardgame-m3", "deck": "NonExistentCard"}).json()["replays"]
            assert len(by_miss) == 0

            # Patch pin + tags + title
            patched = client.patch(
                f"/api/replays/{rp1['id']}",
                json={"pinned": True, "tags": ["eval", "featured"], "title": "Featured Godot Match"},
            )
            assert patched.status_code == 200
            assert patched.json()["replay"]["pinned"] is True
            assert patched.json()["replay"]["tags"] == ["eval", "featured"]

            # Delete copy #2 -> shared blob must stay intact for #1
            assert client.delete(f"/api/replays/{rp2['id']}").status_code == 200
            assert client.get(f"/api/replays/{rp1['id']}/file").content == raw_bytes

            # Delete #1 -> shared blob now unreferenced and deleted from disk
            blob_path = data_dir / "blobs" / "replays" / rp1["blob_sha256"]
            assert blob_path.exists()
            assert client.delete(f"/api/replays/{rp1['id']}").status_code == 200
            assert not blob_path.exists()
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_rl_trace_blunder_detection_and_nan_sanitization(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        record = {
            "version": 2,
            "manifest_hash": "abcd1234",
            "seed1": 7,
            "seed2": 9,
            "deck1": ["Rune Forge", "Anvil Golem"],
            "deck2": ["Storm Caller"],
            "actions": [{"type": "play_card", "player": 1}],
        }
        trace = [
            {
                "turn": 1,
                "player": 1,
                "action_label": "play Rune Forge -> L0",
                "value_before": 0.50,
                "value_after": 0.62,
                "top_k_probs": [{"label": "play Rune Forge -> L0", "prob": 0.72, "chosen": True}],
            },
            {
                "turn": 7,
                "player": 1,
                "action_label": "end_turn (blunder)",
                "value_before": 0.78,
                "value_after": 0.14,
                "debug_non_finite": float("nan"),
                "top_k_probs": [
                    {"label": "end_turn (blunder)", "prob": 0.45, "chosen": True},
                    {"label": "attack lethal", "prob": 0.44},
                ],
            },
            {
                "turn": 9,
                "player": 1,
                "action_label": "concede",
                "value_before": 0.14,
                "value_after": 0.05,
                "inf_probe": float("inf"),
            },
        ]

        # Build JSON containing literal NaN/Infinity tokens to verify server handles & strips non-finite values
        raw_trace_str = json.dumps(trace, allow_nan=True)
        assert "NaN" in raw_trace_str

        with httpx.Client(base_url=url) as client:
            resp = client.post(
                "/api/replays",
                data={
                    "project": "trace-proj",
                    "title": "Blunder Game",
                    "rl_trace_json": raw_trace_str,
                },
                files={"file": ("blunder.json", json.dumps(record).encode("utf-8"), "application/json")},
            )
            assert resp.status_code == 200
            replay = resp.json()["replay"]
            assert replay["has_rl_trace"] is True
            assert replay["blunder_turn"] == 7
            assert replay["max_value_swing"] == pytest.approx(0.64, abs=1e-6)

            filtered = client.get(
                "/api/replays",
                params={"project": "trace-proj", "has_trace": "true", "min_swing": "0.5", "sort_by": "max_value_swing"},
            ).json()["replays"]
            assert len(filtered) == 1
            assert filtered[0]["id"] == replay["id"]

            tr_resp = client.get(f"/api/replays/{replay['id']}/trace")
            assert tr_resp.status_code == 200
            assert "NaN" not in tr_resp.text and "Infinity" not in tr_resp.text
            payload = json.loads(tr_resp.text)
            json.dumps(payload, allow_nan=False)
            assert payload["blunder_turn"] == 7
            assert payload["trace"][1]["debug_non_finite"] is None
            assert payload["trace"][2]["inf_probe"] is None
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_mp4_auto_detection_and_inline_streaming(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        # Mock ISO BMFF ftyp box header for MP4 without specifying .mp4 or format param
        fake_mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"
        rp = tl.log_replay(
            fake_mp4,
            project="media-proj",
            title="Highlight Clip",
            filename="clip_unknown_ext.bin",
            server_url=url,
        )
        assert rp["format"] == "mp4"
        assert rp["content_type"] == "video/mp4"

        with httpx.Client(base_url=url) as client:
            r = client.get(f"/api/replays/{rp['id']}/file", params={"inline": "true"})
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("video/mp4")
            assert r.headers["content-disposition"].startswith("inline")
            assert r.content == fake_mp4
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_card_stats_trends_exports_and_distributions(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        with tl.init(
            project="cards-proj",
            run_id="run_cards_1",
            name="cards-ppo",
            server_url=url,
            spool_dir=tmp_path / "spool",
        ) as run:
            run.log_card_stats(
                [
                    {
                        "card_name": "Firebolt",
                        "played": 40,
                        "drawn": 50,
                        "wins_when_played": 28,
                        "games_when_played": 40,
                        "play_rate": 0.80,
                        "win_rate": 0.70,
                    },
                    {
                        "card_name": "Light Cleric",
                        "played": 20,
                        "drawn": 50,
                        "wins_when_played": 8,
                        "games_when_played": 20,
                        "play_rate": float("nan"),  # verify sanitization
                    },
                ],
                step=10_000,
            )

            # Vector counts + card_names interface
            run.log_card_stats(
                counts=[60, 25],
                card_names=["Firebolt", "Light Cleric"],
                drawn=[70, 50],
                wins_when_played=[36, 10],
                games_when_played=[60, 25],
                step=20_000,
            )

            # Log 12 steps of discrete policy distribution to test entropy & limit downsampling
            for i in range(12):
                run.log_distribution(
                    "policy/action_types",
                    {"play_card": 0.4 + 0.01 * i, "attack": 0.3, "end_turn": 0.3 - 0.01 * i},
                    step=(i + 1) * 1000,
                )

        with httpx.Client(base_url=url) as client:
            meta_res = client.get("/api/cards/meta", params={"run_id": "run_cards_1"})
            assert meta_res.status_code == 200
            meta_json = json.loads(meta_res.text)
            json.dumps(meta_json, allow_nan=False)
            cards = {c["card_name"]: c for c in meta_json["cards"]}
            assert cards["Firebolt"]["played"] == 100
            assert cards["Firebolt"]["drawn"] == 120
            assert cards["Firebolt"]["win_rate"] == pytest.approx(64 / 100)
            assert cards["Light Cleric"]["played"] == 45
            assert cards["Light Cleric"]["win_rate"] == pytest.approx(18 / 45)

            trends_res = client.get("/api/cards/trends", params={"run_id": "run_cards_1"})
            assert trends_res.status_code == 200
            trends = trends_res.json()["trends"]
            assert len(trends["Firebolt"]) == 2
            assert trends["Firebolt"][0]["step"] == 10_000
            assert trends["Firebolt"][1]["step"] == 20_000

            csv_res = client.get("/api/cards/export", params={"run_id": "run_cards_1", "format": "csv"})
            assert csv_res.status_code == 200
            parsed_csv = list(csv.DictReader(io.StringIO(csv_res.text)))
            assert len(parsed_csv) == 4

            json_exp = client.get("/api/cards/export", params={"run_id": "run_cards_1", "format": "json"})
            assert json_exp.status_code == 200
            json.dumps(json.loads(json_exp.text), allow_nan=False)

            dist_res = client.get(
                "/api/distributions",
                params={"run_id": "run_cards_1", "name": "policy/action_types", "limit": "5"},
            )
            assert dist_res.status_code == 200
            dist_data = dist_res.json()
            json.dumps(dist_data, allow_nan=False)
            assert "policy/action_types" in dist_data["names"]
            assert len(dist_data["series"]) == 5
            assert dist_data["series"][0]["step"] == 1000
            assert dist_data["series"][-1]["step"] == 12000
            assert math.isfinite(dist_data["series"][0]["entropy"])
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

"""Comprehensive tests for Card Meta analytics, Dual WPA, Ridge Impact, Synergy, 4-Lane Preference, and UI routes."""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
import shutil
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
from traininglog.server.db import Database


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


def test_meta_gzip_batch_idempotency_and_nan_sanitization(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        games = [
            {
                "source_key": "g_idemp_1",
                "source": "league",
                "manifest_hash": "m_v1",
                "rounds": 6,
                "critic_perspective": "reference_public",
                "players": [
                    {
                        "seat": 0,
                        "player_key": "AgentA",
                        "actor_type": "ai",
                        "rating": 1700.0,
                        "rank": 1,
                        "outcome_score": 1.0,
                        "deck": {"Firebolt": 2, "Shield": 2},
                        "opening_hand": ["Firebolt"],
                        "drawn": {"Firebolt": 1},
                        "draw_wpas": {"Firebolt": [0.05, float("nan")]},
                        "plays": [
                            {
                                "card": "Firebolt",
                                "turn": 2,
                                "lane": 1,
                                "lane_context": "opposed",
                                "win_prob_before": 0.50,
                                "win_prob_after": 0.62,
                            }
                        ],
                    },
                    {
                        "seat": 1,
                        "player_key": "AgentB",
                        "actor_type": "ai",
                        "rating": 1650.0,
                        "rank": 2,
                        "outcome_score": 0.0,
                        "deck": {"Firebolt": 2, "Shield": 2},
                        "drawn": {"Shield": 1},
                        "plays": [
                            {
                                "card": "Shield",
                                "turn": 2,
                                "lane": 0,
                                "win_prob_before": float("inf"),
                                "win_prob_after": 0.40,
                            }
                        ],
                    },
                ],
            }
        ]
        # Send raw gzip payload containing literal NaN/Inf to ensure server sanitizes cleanly
        raw_payload = gzip.compress(
            json.dumps({"project": "meta-nan", "games": games}, allow_nan=True).encode("utf-8")
        )
        with httpx.Client(base_url=url) as client:
            r1 = client.post(
                "/api/meta/games/batch",
                content=raw_payload,
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
            )
            assert r1.status_code == 200
            assert r1.json()["inserted"] == 1
            assert r1.json()["duplicates"] == 0

            # Duplicate batch with same source_key
            r2 = client.post(
                "/api/meta/games/batch",
                content=raw_payload,
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
            )
            assert r2.status_code == 200
            assert r2.json()["inserted"] == 0
            assert r2.json()["duplicates"] == 1

            ov = client.get("/api/meta/overview", params={"project": "meta-nan", "min_games": 0})
            assert ov.status_code == 200
            assert "NaN" not in ov.text and "Infinity" not in ov.text
            cards = {c["card_key"]: c for c in ov.json()["cards"]}
            assert cards["Firebolt"]["critic_play_wpa"] == pytest.approx(0.12, abs=1e-6)
            assert cards["Firebolt"]["critic_draw_wpa"] == pytest.approx(0.05, abs=1e-6)
            assert cards["Shield"]["critic_play_wpa"] is None
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_meta_confounded_ridge_regression_recovers_true_card_impact(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        rng = random.Random(123)
        games = []
        # StrongPlayer (rating 2100) always plays WeakTrap; LowPlayer (rating 1300) always plays HiddenGem.
        # Skill gap (~800 rating -> +4.0 logit) swamps card effect in raw WR,
        # so WeakTrap wins ~75% raw while HiddenGem loses ~75% raw.
        # Within matched-rating games and overall, HiddenGem gives +2.0 logit while WeakTrap gives -1.0 logit.
        for i in range(180):
            if i < 100:
                r0, r1 = 2100.0, 1300.0
                c0, c1 = "WeakTrap", "HiddenGem"
                # Skill gap (+4.0 logit) dominates card strength (-1.5), so WeakTrap wins ~90% raw
                p0_logit = 2.3
            elif i < 140:
                # Equal-rating games expose that HiddenGem beats WeakTrap
                r0, r1 = 1700.0, 1700.0
                c0, c1 = "HiddenGem", "WeakTrap"
                p0_logit = 2.0
            else:
                r0, r1 = 1700.0, 1700.0
                c0, c1 = "NeutralCard", "WeakTrap"
                p0_logit = 1.0

            import math
            p0_win = rng.random() < (1.0 / (1.0 + math.exp(-p0_logit)))
            games.append(
                {
                    "source_key": f"conf:{i}",
                    "manifest_hash": "p1",
                    "rounds": 6,
                    "players": [
                        {
                            "seat": 0,
                            "player_key": f"P0_{int(r0)}",
                            "actor_type": "ai",
                            "rating": r0,
                            "rank": 1 if p0_win else 2,
                            "outcome_score": 1.0 if p0_win else 0.0,
                            "deck": [c0, "FillerA", "FillerB"],
                            "drawn": [c0, "FillerA"],
                            "plays": [{"card": c0, "turn": 2, "lane": 0}],
                        },
                        {
                            "seat": 1,
                            "player_key": f"P1_{int(r1)}",
                            "actor_type": "ai",
                            "rating": r1,
                            "rank": 2 if p0_win else 1,
                            "outcome_score": 0.0 if p0_win else 1.0,
                            "deck": [c1, "FillerA", "FillerB"],
                            "drawn": [c1, "FillerA"],
                            "plays": [{"card": c1, "turn": 2, "lane": 1}],
                        },
                    ],
                }
            )

        tl.log_meta_games(games, project="conf-proj", server_url=url)
        with httpx.Client(base_url=url) as client:
            res = client.get("/api/meta/overview", params={"project": "conf-proj", "min_games": 0}).json()
            cards = {c["card_key"]: c for c in res["cards"]}
            # Raw WR is confounded in favor of WeakTrap
            assert cards["WeakTrap"]["gih_wr"] > cards["HiddenGem"]["gih_wr"]
            # Regression isolated impact recovers HiddenGem > WeakTrap
            assert cards["HiddenGem"]["isolated_impact"] > cards["WeakTrap"]["isolated_impact"]
            assert cards["HiddenGem"]["isolated_impact"] > 0
            assert cards["WeakTrap"]["isolated_impact"] < 0
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_meta_dual_wpa_mirror_exclusion_and_critic_perspectives(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        games = []
        # 20 non-mirror games: P0 draws AceCard & wins 18/20; P1 doesn't have AceCard.
        # Also 20 games where P0 has AceCard in deck but does NOT draw it & wins 8/20.
        # Also 20 mirror games where BOTH draw AceCard (naturally 50% WR -> compresses raw WR & WPA).
        for i in range(20):
            p0_win = i < 18
            games.append(
                {
                    "source_key": f"nm_win:{i}",
                    "manifest_hash": "m1",
                    "critic_perspective": "reference_public",
                    "players": [
                        {
                            "seat": 0, "player_key": "A", "rating": 1600,
                            "outcome_score": 1.0 if p0_win else 0.0,
                            "deck": ["AceCard", "Filler"],
                            "drawn": ["AceCard"],
                            "plays": [{"card": "AceCard", "turn": 2, "lane": 0, "win_prob_before": 0.50, "win_prob_after": 0.65}],
                        },
                        {
                            "seat": 1, "player_key": "B", "rating": 1600,
                            "outcome_score": 0.0 if p0_win else 1.0,
                            "deck": ["Filler"],
                            "drawn": ["Filler"],
                        },
                    ],
                }
            )
        for i in range(20):
            p0_win = i < 8
            games.append(
                {
                    "source_key": f"gns:{i}",
                    "manifest_hash": "m1",
                    "critic_perspective": "actor",  # different perspective, must NOT mix with reference_public
                    "players": [
                        {
                            "seat": 0, "player_key": "A", "rating": 1600,
                            "outcome_score": 1.0 if p0_win else 0.0,
                            "deck": ["AceCard", "Filler"],
                            "drawn": ["Filler"],
                            "plays": [{"card": "Filler", "turn": 2, "lane": 0, "win_prob_before": 0.50, "win_prob_after": 0.99}],
                        },
                        {
                            "seat": 1, "player_key": "B", "rating": 1600,
                            "outcome_score": 0.0 if p0_win else 1.0,
                            "deck": ["Filler"],
                            "drawn": ["Filler"],
                        },
                    ],
                }
            )
        for i in range(20):
            p0_win = i % 2 == 0
            games.append(
                {
                    "source_key": f"mirror:{i}",
                    "manifest_hash": "m1",
                    "critic_perspective": "reference_public",
                    "players": [
                        {
                            "seat": 0, "player_key": "A", "rating": 1600,
                            "outcome_score": 1.0 if p0_win else 0.0,
                            "deck": ["AceCard", "Filler"],
                            "drawn": ["AceCard"],
                        },
                        {
                            "seat": 1, "player_key": "B", "rating": 1600,
                            "outcome_score": 0.0 if p0_win else 1.0,
                            "deck": ["AceCard", "Filler"],
                            "drawn": ["AceCard"],
                        },
                    ],
                }
            )

        tl.log_meta_games(games, project="wpa-proj", server_url=url)
        with httpx.Client(base_url=url) as client:
            inc_m = client.get(
                "/api/meta/overview",
                params={"project": "wpa-proj", "exclude_mirrors": "false", "min_games": 0},
            ).json()
            exc_m = client.get(
                "/api/meta/overview",
                params={"project": "wpa-proj", "exclude_mirrors": "true", "min_games": 0},
            ).json()

            assert inc_m["kpis"]["critic_perspective"] == "reference_public"
            ace_inc = next(c for c in inc_m["cards"] if c["card_key"] == "AceCard")
            ace_exc = next(c for c in exc_m["cards"] if c["card_key"] == "AceCard")

            # Without mirror exclusion: (18 + 20) / 60 = 63.33% GIH WR
            assert ace_inc["gih_wr"] == pytest.approx(38 / 60, abs=1e-4)
            # With mirror exclusion: 18 / 20 = 90% GIH WR, GNS = 8 / 20 = 40%, raw drawn WPA = +50%
            assert ace_exc["gih_wr"] == pytest.approx(0.90, abs=1e-4)
            assert ace_exc["gns_wr"] == pytest.approx(0.40, abs=1e-4)
            assert ace_exc["drawn_wpa_raw"] == pytest.approx(0.50, abs=1e-4)
            assert ace_exc["drawn_wpa"] > ace_inc["drawn_wpa"]
            # Critic play WPA stays isolated to reference_public (+0.15) and ignores actor Filler (+0.49)
            assert ace_inc["critic_play_wpa"] == pytest.approx(0.15, abs=1e-4)
            filler = next(c for c in inc_m["cards"] if c["card_key"] == "Filler")
            assert filler["critic_play_wpa"] is None
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_meta_synergy_counters_and_4lane_misplay_detection(tmp_path: Path) -> None:
    url, server, thread = _start_server(tmp_path / "data")
    try:
        rng = random.Random(77)
        games = []
        # ComboA + ComboB win 90% when played in SAME LANE, 50% otherwise.
        # StrongA + StrongB are individually strong (~65%) with NO extra combo lift.
        # MisplayCard is played 75% in L0 (35% WR) and 25% in L3 (85% WR).
        # GlobalSpell is played with lane=None and lane_context=None.
        # TargetCard loses badly when opponent draws HardCounter.
        for i in range(160):
            use_combo = i < 50
            use_strong_pair = 50 <= i < 100
            use_misplay = True
            l_mis = 0 if (i % 4 != 0) else 3
            opp_has_counter = i >= 110

            if use_combo:
                p0_cards = ["ComboA", "ComboB", "MisplayCard", "GlobalSpell"]
                p0_plays = [
                    {"card": "ComboA", "turn": 2, "lane": 1, "lane_context": "empty"},
                    {"card": "ComboB", "turn": 3, "lane": 1, "lane_context": "allied"},
                    {"card": "MisplayCard", "turn": 4, "lane": l_mis, "lane_context": "opposed" if l_mis == 3 else "empty"},
                    {"card": "GlobalSpell", "turn": 5, "lane": None},
                ]
                base_p0_win = 0.90 if l_mis == 3 else 0.78
            elif use_strong_pair:
                p0_cards = ["StrongA", "StrongB", "MisplayCard", "TargetCard"]
                p0_plays = [
                    {"card": "StrongA", "turn": 2, "lane": 0},
                    {"card": "StrongB", "turn": 3, "lane": 2},
                    {"card": "MisplayCard", "turn": 4, "lane": l_mis, "lane_context": "opposed" if l_mis == 3 else "empty"},
                ]
                base_p0_win = 0.62 if l_mis == 3 else 0.56
            else:
                c_pick = ["ComboA"] if i % 2 == 0 else ["ComboB"]
                p0_cards = c_pick + ["MisplayCard", "TargetCard"]
                p0_plays = [
                    {"card": c_pick[0], "turn": 2, "lane": 2},
                    {"card": "MisplayCard", "turn": 4, "lane": l_mis, "lane_context": "opposed" if l_mis == 3 else "empty"},
                ]
                base_p0_win = 0.72 if l_mis == 3 else 0.28

            if opp_has_counter:
                base_p0_win = 0.12

            p0_win = rng.random() < base_p0_win
            # HardCounter also appears against non-TargetCard decks (0..49) with normal WR so its baseline WR isn't 88%
            p1_drawn = ["HardCounter", "P1Filler"] if (opp_has_counter or i < 50) else ["P1Filler"]
            games.append(
                {
                    "source_key": f"syn:{i}",
                    "manifest_hash": "m1",
                    "rounds": 6,
                    "players": [
                        {
                            "seat": 0, "player_key": "A", "rating": 1600,
                            "outcome_score": 1.0 if p0_win else 0.0,
                            "deck": p0_cards + ["Filler"],
                            "drawn": p0_cards,
                            "plays": p0_plays,
                        },
                        {
                            "seat": 1, "player_key": "B", "rating": 1600,
                            "outcome_score": 0.0 if p0_win else 1.0,
                            "deck": p1_drawn + ["Filler"],
                            "drawn": p1_drawn,
                            "plays": [{"card": p1_drawn[0], "turn": 2, "lane": 0}],
                        },
                    ],
                }
            )

        tl.log_meta_games(games, project="syn-proj", server_url=url)
        with httpx.Client(base_url=url) as client:
            ov = client.get("/api/meta/overview", params={"project": "syn-proj", "min_games": 0}).json()
            cards = {c["card_key"]: c for c in ov["cards"]}

            # ComboA's top same-lane partner should be ComboB
            assert cards["ComboA"]["top_same_lane_synergy"]["partner"] == "ComboB"
            assert cards["ComboA"]["top_same_lane_synergy"]["lift"] > 0.08

            # MisplayCard favors L0, performs best in L3, and triggers has_misplay_gap
            mp = cards["MisplayCard"]["lane_preference"]
            assert mp["favored_lane"] == 0
            assert mp["best_lane"] == 3
            assert mp["has_misplay_gap"] is True
            assert mp["concentration"] > 0.1

            # GlobalSpell has lane_plays == 0 and global_plays > 0
            gp = cards["GlobalSpell"]["lane_preference"]
            assert gp["lane_plays"] == 0
            assert gp["global_plays"] == 50

            # Synergy & Counter endpoints
            syn_lane = client.get("/api/meta/synergies", params={"project": "syn-proj", "level": "same_lane"}).json()
            top_pair = syn_lane["positive"][0]
            assert {top_pair["card_a"], top_pair["card_b"]} == {"ComboA", "ComboB"}

            detail = client.get("/api/meta/cards/TargetCard", params={"project": "syn-proj"}).json()
            worst_ctr = detail["counters"]["worst_against"][0]
            assert worst_ctr["opponent"] == "HardCounter"
            assert worst_ctr["lift"] < 0

            mp_detail = client.get("/api/meta/cards/MisplayCard", params={"project": "syn-proj"}).json()
            ctx_names = {c["context"] for c in mp_detail["lane_contexts"]}
            assert {"empty", "opposed"}.issubset(ctx_names)
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_meta_ffa_bot_exclusion_sources_patch_diff_cache_and_prune(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    url, server, thread = _start_server(data_dir)
    try:
        # Catalogs for v1 and v2
        tl.log_card_catalog(
            [{"card_key": "Bolt", "display_name": "Bolt", "mana_cost": 2, "faction": "Red", "card_type": "Spell"}],
            project="multi-proj",
            manifest_hash="v1",
            patch_label="Patch 1.0",
            server_url=url,
        )
        tl.log_card_catalog(
            [{"card_key": "Bolt", "display_name": "Bolt", "mana_cost": 3, "faction": "Red", "card_type": "Spell"}],
            project="multi-proj",
            manifest_hash="v2",
            patch_label="Patch 1.1",
            server_url=url,
        )

        day_ms = 86_400_000
        now_ms = int(time.time() * 1000)

        # League games on v1 & v2 + Bot game + Player Ranked + 4-player FFA
        games = []
        for i in range(12):
            games.append(
                {
                    "source_key": f"lg_v1:{i}",
                    "source": "league",
                    "manifest_hash": "v1",
                    "played_at": now_ms - 10 * day_ms,
                    "players": [
                        {"seat": 0, "player_key": "A", "actor_type": "ai", "rating": 1800, "outcome_score": 1.0 if i < 9 else 0.0, "deck": ["Bolt"], "drawn": ["Bolt"], "plays": [{"card": "Bolt", "turn": 2, "lane": 0}]},
                        {"seat": 1, "player_key": "B", "actor_type": "ai", "rating": 1800, "outcome_score": 0.0 if i < 9 else 1.0, "deck": ["Shield"], "drawn": ["Shield"], "plays": [{"card": "Shield", "turn": 2, "lane": 1}]},
                    ],
                }
            )
            games.append(
                {
                    "source_key": f"lg_v2:{i}",
                    "source": "league",
                    "manifest_hash": "v2",
                    "played_at": now_ms - 1 * day_ms,
                    "players": [
                        {"seat": 0, "player_key": "A", "actor_type": "ai", "rating": 1800, "outcome_score": 1.0 if i < 4 else 0.0, "deck": ["Bolt"], "drawn": ["Bolt"], "plays": [{"card": "Bolt", "turn": 3, "lane": 0}]},
                        {"seat": 1, "player_key": "B", "actor_type": "ai", "rating": 1800, "outcome_score": 0.0 if i < 4 else 1.0, "deck": ["Shield"], "drawn": ["Shield"], "plays": [{"card": "Shield", "turn": 2, "lane": 1}]},
                    ],
                }
            )
        # Bot game
        games.append(
            {
                "source_key": "bot_game:1",
                "source": "league",
                "manifest_hash": "v2",
                "played_at": now_ms - 1 * day_ms,
                "players": [
                    {"seat": 0, "player_key": "A", "actor_type": "ai", "rating": 1800, "outcome_score": 1.0, "deck": ["BotFarmCard"], "drawn": ["BotFarmCard"]},
                    {"seat": 1, "player_key": "Random_Bot", "actor_type": "bot", "rating": 1000, "outcome_score": 0.0, "deck": ["Shield"], "drawn": ["Shield"]},
                ],
            }
        )
        # Human player_ranked game
        games.append(
            {
                "source_key": "human:1",
                "source": "player_ranked",
                "manifest_hash": "v2",
                "played_at": now_ms - 1 * day_ms,
                "players": [
                    {"seat": 0, "player_key": "h1", "actor_type": "human", "rating": 1500, "tier": "Gold", "outcome_score": 1.0, "deck": ["HumanCard"], "drawn": ["HumanCard"]},
                    {"seat": 1, "player_key": "h2", "actor_type": "human", "rating": 1500, "tier": "Gold", "outcome_score": 0.0, "deck": ["Shield"], "drawn": ["Shield"]},
                ],
            }
        )
        # 4-player FFA game (rank omitted -> auto-computed outcome_score from rank)
        for i in range(6):
            games.append(
                {
                    "source_key": f"ffa:{i}",
                    "source": "league",
                    "mode": "multiplayer",
                    "manifest_hash": "v2",
                    "played_at": now_ms - 1 * day_ms,
                    "players": [
                        {"seat": 0, "player_key": "A", "rank": 1, "deck": ["FfaWinner"], "drawn": ["FfaWinner"]},
                        {"seat": 1, "player_key": "B", "rank": 2, "deck": ["Shield"], "drawn": ["Shield"]},
                        {"seat": 2, "player_key": "C", "rank": 3, "deck": ["Shield"], "drawn": ["Shield"]},
                        {"seat": 3, "player_key": "D", "rank": 4, "deck": ["Shield"], "drawn": ["Shield"]},
                    ],
                }
            )

        # Test NDJSON importer for the last batch too
        ndjson_path = tmp_path / "extra_games.ndjson.gz"
        with gzip.open(ndjson_path, "wt", encoding="utf-8") as fh:
            for g in games:
                fh.write(json.dumps(g) + "\n")

        imp_res = tl.import_meta_games_ndjson(ndjson_path, project="multi-proj", server_url=url)
        assert imp_res["inserted"] == len(games)

        with httpx.Client(base_url=url) as client:
            # Exclude bots drops BotFarmCard
            ov_no_bot = client.get(
                "/api/meta/overview",
                params={"project": "multi-proj", "source": "league", "manifest_hash": "v2", "exclude_bots": "true", "min_games": 0},
            ).json()
            keys_no_bot = {c["card_key"] for c in ov_no_bot["cards"]}
            assert "BotFarmCard" not in keys_no_bot
            assert "HumanCard" not in keys_no_bot

            ov_with_bot = client.get(
                "/api/meta/overview",
                params={"project": "multi-proj", "source": "league", "manifest_hash": "v2", "exclude_bots": "false", "min_games": 0},
            ).json()
            assert "BotFarmCard" in {c["card_key"] for c in ov_with_bot["cards"]}

            # Player ranked source separation
            ov_human = client.get(
                "/api/meta/overview",
                params={"project": "multi-proj", "source": "player_ranked", "manifest_hash": "v2", "tier": "Gold", "min_games": 0},
            ).json()
            assert "HumanCard" in {c["card_key"] for c in ov_human["cards"]}

            # FFA mode
            ov_ffa = client.get(
                "/api/meta/overview",
                params={"project": "multi-proj", "source": "league", "manifest_hash": "v2", "mode": "multiplayer", "min_games": 0},
            ).json()
            ffa_cards = {c["card_key"]: c for c in ov_ffa["cards"]}
            assert ffa_cards["FfaWinner"]["gih_wr"] == pytest.approx(1.0)

            # Patch Diff shows Bolt mana_cost 2->3 and drop in WR
            pdiff = client.get(
                "/api/meta/patch-diff",
                params={"project": "multi-proj", "source": "league", "manifest_a": "v1", "manifest_b": "v2"},
            ).json()
            bolt_diff = next(d for d in pdiff["diffs"] if d["card_key"] == "Bolt")
            assert any(ch["field"] == "mana_cost" and ch["old"] == 2 and ch["new"] == 3 for ch in bolt_diff["catalog_changes"])
            assert bolt_diff["delta_gih_wr"] < 0

            # Trends & CSV export
            trends = client.get("/api/meta/trends", params={"project": "multi-proj", "source": "league", "manifest_hash": "all"}).json()
            assert len(trends["trends"]["Bolt"]) >= 2

            csv_exp = client.get("/api/meta/export", params={"project": "multi-proj", "source": "league", "manifest_hash": "v2", "min_games": 0, "format": "csv"})
            assert csv_exp.status_code == 200
            rows = list(csv.DictReader(io.StringIO(csv_exp.text)))
            assert any(r["card_key"] == "Bolt" for r in rows)

            # Watermark invalidation when ingesting another game
            tl.log_meta_games(
                [
                    {
                        "source_key": "lg_v2:extra",
                        "source": "league",
                        "manifest_hash": "v2",
                        "players": [
                            {"seat": 0, "player_key": "A", "outcome_score": 1.0, "deck": ["NewCard"], "drawn": ["NewCard"]},
                            {"seat": 1, "player_key": "B", "outcome_score": 0.0, "deck": ["Shield"], "drawn": ["Shield"]},
                        ],
                    }
                ],
                project="multi-proj",
                server_url=url,
            )
            ov_after = client.get(
                "/api/meta/overview",
                params={"project": "multi-proj", "source": "league", "manifest_hash": "v2", "min_games": 0},
            ).json()
            assert "NewCard" in {c["card_key"] for c in ov_after["cards"]}
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

    # Prune old meta_card_plays (> 5 days old) -> deletes v1 plays while preserving v1 meta_game_cards!
    db = Database(data_dir / "traininglog.db")
    try:
        prune_res = db.prune_meta_card_plays(project="multi-proj", older_than_days=5)
        assert prune_res["deleted_plays"] == 24
        # v1 Bolt funnel counts still preserved in meta_game_cards
        from traininglog.server.meta import MetaAnalyticsEngine, MetaFilter
        eng = MetaAnalyticsEngine(db)
        ov_v1 = eng.get_overview(MetaFilter(project="multi-proj", source="league", manifest_hash="v1", min_games=0))
        bolt_v1 = next(c for c in ov_v1["cards"] if c["card_key"] == "Bolt")
        assert bolt_v1["games_in_deck"] == 12
        assert bolt_v1["gih_wr"] == pytest.approx(0.75)
    finally:
        db.close()


def test_existing_db_migration_safety(tmp_path: Path) -> None:
    src_db = Path(__file__).resolve().parents[1] / "data" / "traininglog.db"
    if not src_db.exists():
        pytest.skip("No pre-existing data/traininglog.db to test migration against")
    copy_db = tmp_path / "traininglog.db"
    shutil.copy2(src_db, copy_db)
    db = Database(copy_db)
    try:
        assert db.get_meta_watermark(None) == 0
        assert db.list_projects()
    finally:
        db.close()

"""Realistic simulator for cardgame PPO runs (`mlp`, `transformer`, `dual_stream`)."""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from collections import Counter

import numpy as np

import traininglog as tl


def _simulate_single_run(
    *,
    server_url: str,
    project: str,
    arch: str,
    lr: float,
    ent_coef: float,
    target_kl: float,
    anneal_dense: bool,
    updates: int,
    step_delta: int,
    delay: float,
    inject_anomaly: bool,
    seed: int,
    run_ids_out: dict[str, str] | None = None,
) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    total_timesteps = updates * step_delta
    run_name = f"cardgame-v0__{arch}__ppo__{seed}__{int(time.time())}"
    config = {
        "gym_id": "cardgame-v0",
        "arch": arch,
        "learning_rate": lr,
        "ent_coef": ent_coef,
        "target_kl": target_kl,
        "num_envs": 4096,
        "num_steps": 64,
        "num_minibatches": 8 if arch != "mlp" else 4,
        "update_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "anneal_dense_rewards": anneal_dense,
        "dense_reward_scale_initial": 1.0,
        "dense_reward_scale_final": 0.0 if anneal_dense else 1.0,
        "agent_player": "balanced",
        "opponent_pool": "random:15,heuristic:25,selfplay:30,past_models:15,saved_models:15",
        "amp": True,
        "seed": seed,
    }

    with tl.init(
        project=project,
        name=run_name,
        group=f"{arch}-league",
        tags=[arch, "ppo", "annealed" if anneal_dense else "dense"],
        config=config,
        arch=arch,
        obs_dim=58,
        manifest_hash="c4f8a91e",
        total_timesteps=total_timesteps,
        server_url=server_url,
        capture_console=True,
        capture_system=True,
        system_interval=2.0 if delay > 0.02 else 10.0,
    ) as run:
        writer = run.summary_writer()
        if run_ids_out is not None:
            run_ids_out[arch] = run.id
        card_counts: Counter[str] = Counter()
        card_names = [
            "Mica_Anvil_Bringer",
            "Astos_Blade_Unending",
            "Tuff_Shieldbearer",
            "Dunite_Sword_of_Steel",
            "Scoria_The_Deadshot",
            "Drop_The_Hammer",
        ]

        arch_bonus = {"dual_stream": 0.14, "transformer": 0.10, "mlp": 0.02}.get(arch, 0.05)

        for upd in range(1, updates + 1):
            for cmd in run.poll_commands():
                print(f"[{run_name}] Received cooperative command: {cmd.type}")
                if cmd.type == "checkpoint":
                    cmd.complete(f"Saved checkpoint at step {upd * step_delta:,}")
                elif cmd.type in ("eval", "run_redo"):
                    cmd.complete(f"Executed {cmd.type} at update {upd}")
                elif cmd.type == "stop":
                    cmd.complete("Stopping gracefully on user request")
                    break

            if run.should_stop:
                break

            global_step = upd * step_delta
            frac = upd / max(1, updates)
            dense_scale = max(0.0, 1.0 - frac) if anneal_dense else 1.0
            cur_lr = lr * (1.0 - 0.85 * frac)

            # Realistic learning curves
            win_overall = min(
                0.92,
                max(0.18, 0.36 + arch_bonus + 0.34 * (1.0 - math.exp(-3.2 * frac)) + rng.uniform(-0.02, 0.02)),
            )
            win_p1 = min(0.96, win_overall + 0.028 + rng.uniform(-0.01, 0.01))
            win_p2 = max(0.12, win_overall - 0.028 + rng.uniform(-0.01, 0.01))

            sps_base = {"mlp": 420_000, "transformer": 210_000, "dual_stream": 175_000}.get(arch, 200_000)
            current_sps = int(sps_base + rng.randint(-12_000, 12_000))

            p_loss = max(0.01, 0.22 * math.exp(-2.1 * frac) + rng.uniform(-0.01, 0.01))
            v_loss = max(0.02, 0.65 * math.exp(-2.5 * frac) + rng.uniform(-0.02, 0.02))
            ent_hand = max(0.12, 1.45 * math.exp(-1.1 * frac) + rng.uniform(-0.03, 0.03))
            ent_board = max(0.08, 0.95 * math.exp(-1.0 * frac) + rng.uniform(-0.02, 0.02))
            ent_total = ent_hand + ent_board
            approx_kl = max(0.002, min(target_kl * 0.9, 0.014 + rng.uniform(-0.004, 0.005)))

            if inject_anomaly and arch == "mlp" and upd == max(6, updates // 2):
                approx_kl = 0.28  # triggers kl_spike
            if inject_anomaly and arch == "mlp" and upd == max(8, updates // 2 + 2):
                v_loss = float("nan")  # triggers nan_inf + red vertical marker

            writer.add_scalar("charts/learning_rate", cur_lr, global_step)
            writer.add_scalar("charts/dense_reward_scale", dense_scale, global_step)
            writer.add_scalar("charts/episodic_return", -1.5 + 8.5 * win_overall + rng.uniform(-0.2, 0.2), global_step)
            writer.add_scalar("charts/episodic_length", 34.0 - 8.0 * frac + rng.uniform(-1.0, 1.0), global_step)
            writer.add_scalar("charts/SPS", current_sps, global_step)

            writer.add_scalar("losses/value_loss", v_loss, global_step)
            writer.add_scalar("losses/policy_loss", p_loss, global_step)
            writer.add_scalar("losses/entropy", ent_total, global_step)
            writer.add_scalar("losses/entropy_hand", ent_hand, global_step)
            writer.add_scalar("losses/entropy_board", ent_board, global_step)
            writer.add_scalar("losses/approx_kl", approx_kl, global_step)
            writer.add_scalar("losses/clipfrac", 0.08 + rng.uniform(-0.02, 0.02), global_step)
            writer.add_scalar("losses/explained_variance", min(0.95, 0.25 + 0.65 * (1 - math.exp(-3 * frac))), global_step)

            # Winrate overall, seat splits & opponent pool
            writer.add_scalar("game/win_rate_overall", win_overall, global_step)
            writer.add_scalar("game/win_rate_p1", win_p1, global_step)
            writer.add_scalar("game/win_rate_p2", win_p2, global_step)
            writer.add_scalar("charts/winrate_random", min(0.99, win_overall + 0.25), global_step)
            writer.add_scalar("charts/winrate_heuristic", min(0.94, win_overall + 0.08), global_step)
            writer.add_scalar("charts/winrate_bot", min(0.96, win_overall + 0.14), global_step)
            writer.add_scalar("charts/winrate_selfplay", 0.50 + rng.uniform(-0.03, 0.03), global_step)
            writer.add_scalar("charts/winrate_past_models", min(0.85, 0.52 + 0.15 * frac), global_step)
            writer.add_scalar("charts/winrate_saved_models", max(0.15, win_overall - 0.12), global_step)

            # Lane placement rates & invalid/pass
            lane_raw = [0.29, 0.26, 0.24, 0.21]
            for li, lr_val in enumerate(lane_raw):
                writer.add_scalar(f"actions/lane_{li}_rate", lr_val + rng.uniform(-0.015, 0.015), global_step)
            writer.add_scalar("actions/episode_pass_rate", max(0.05, 0.22 - 0.10 * frac), global_step)
            writer.add_scalar("actions/invalid_action_rate", max(0.001, 0.06 * math.exp(-4.0 * frac)), global_step)

            # Board economy & rewards
            writer.add_scalar("game/mana_efficiency", min(0.96, 0.55 + 0.35 * frac), global_step)
            writer.add_scalar("game/trade_ratio", 0.9 + 0.6 * win_overall, global_step)
            writer.add_scalar("game/card_advantage", -0.5 + 2.2 * win_overall, global_step)
            writer.add_scalar("game/board_power_margin", -1.2 + 5.0 * win_overall, global_step)
            writer.add_scalar("game/final_hp_margin", -4.0 + 18.0 * win_overall, global_step)
            writer.add_scalar("game/game_length_rounds", 7.2 - 1.4 * frac, global_step)
            writer.add_scalar("game/artifacts_unlocked_avg", min(2.4, 0.4 + 1.8 * frac), global_step)
            writer.add_scalar("game/artifact_unlock_rate", min(0.88, 0.25 + 0.55 * frac), global_step)

            writer.add_scalar("rewards/damage", 1.2 * dense_scale, global_step)
            writer.add_scalar("rewards/trades", 0.8 * dense_scale, global_step)
            writer.add_scalar("rewards/mana_waste", -0.35 * dense_scale, global_step)
            writer.add_scalar("rewards/dominance", 0.45 * dense_scale, global_step)
            writer.add_scalar("rewards/kills", 0.7 * dense_scale, global_step)
            writer.add_scalar("rewards/win_loss", (win_overall * 2.0 - 1.0) * 5.0, global_step)

            for cname in card_names:
                c = rng.randint(100, 500)
                card_counts[cname] += c
                writer.add_scalar(f"cards/play_rate_{cname}", c / 1800.0, global_step)

            if upd % max(1, updates // 4) == 0 or upd == updates:
                lane_samples = np_rng.choice([0, 1, 2, 3], size=1024, p=[0.29, 0.26, 0.24, 0.21])
                writer.add_histogram("actions/lane_placement_histogram", lane_samples, global_step, bins=4)
                writer.add_histogram("actions/unit_lane_placement_histogram", lane_samples, global_step, bins=4)

            if upd % max(1, updates // 10) == 0 or upd == 1 or upd == updates:
                print(
                    f"Update {upd:4d}/{updates} | Step {global_step:11,d} | "
                    f"SPS: {current_sps:6,d} | WinRate: {win_overall * 100:5.1f}% "
                    f"(P1: {win_p1 * 100:4.1f}% | P2: {win_p2 * 100:4.1f}%) | "
                    f"Loss: (P: {p_loss:.3f}, V: {v_loss:.3f}) | Ent: {ent_total:.2f}"
                )

            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate cardgame PPO training runs")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--project", default="cardgame-ppo")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--step-delta", type=int, default=25_000_000)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--inject-anomaly", action="store_true")
    args = parser.parse_args()

    delay = 0.01 if args.fast else 0.15

    specs = [
        dict(arch="dual_stream", lr=1.5e-4, ent_coef=0.015, target_kl=0.025, anneal_dense=True, seed=101),
        dict(arch="transformer", lr=2.0e-4, ent_coef=0.02, target_kl=0.03, anneal_dense=True, seed=202),
        dict(arch="mlp", lr=2.5e-4, ent_coef=0.01, target_kl=0.02, anneal_dense=False, seed=303),
    ]

    run_ids_by_arch: dict[str, str] = {}
    threads = [
        threading.Thread(
            target=_simulate_single_run,
            kwargs=dict(
                server_url=args.url,
                project=args.project,
                updates=args.updates,
                step_delta=args.step_delta,
                delay=delay,
                inject_anomaly=args.inject_anomaly,
                run_ids_out=run_ids_by_arch,
                **spec,
            ),
        )
        for spec in specs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _simulate_league_and_plasticity(
        server_url=args.url,
        project=args.project,
        run_ids_by_arch=run_ids_by_arch,
        final_step=args.updates * args.step_delta,
        inject_anomaly=args.inject_anomaly,
    )
    _simulate_replays_and_cards(
        server_url=args.url,
        project=args.project,
        run_ids_by_arch=run_ids_by_arch,
        final_step=args.updates * args.step_delta,
    )

    print("Simulation complete!")


def _simulate_league_and_plasticity(
    *,
    server_url: str,
    project: str,
    run_ids_by_arch: dict[str, str],
    final_step: int,
    inject_anomaly: bool,
) -> None:
    """Populate checkpoints with lineage, a Glicko-2 + OpenSkill league, and ReDo plasticity reports."""
    tl.log_checkpoint("builtin:Random_Bot", project=project, name="Random_Bot", server_url=server_url)
    tl.log_checkpoint("builtin:Heuristic_Bot", project=project, name="Heuristic_Bot", server_url=server_url)
    tl.log_checkpoint("builtin:Advanced_Bot", project=project, name="Advanced_Bot", server_url=server_url)

    tl.log_checkpoint(
        "runs/saved_runs/MlpBaseline_400M.pt",
        project=project,
        name="MlpBaseline_400M",
        step=final_step,
        run_id=run_ids_by_arch.get("mlp"),
        arch="mlp",
        obs_dim=58,
        manifest_hash="c4f8a91e",
        tags=["saved_run", "league"],
        server_url=server_url,
    )
    tl.log_checkpoint(
        "runs/saved_runs/Transformer_550M.pt",
        project=project,
        name="Transformer_550M",
        step=final_step,
        run_id=run_ids_by_arch.get("transformer"),
        parent="MlpBaseline_400M",
        arch="transformer",
        obs_dim=58,
        manifest_hash="c4f8a91e",
        tags=["saved_run", "league"],
        server_url=server_url,
    )
    tl.log_checkpoint(
        "runs/saved_runs/DualStream_v1.pt",
        project=project,
        name="DualStream_v1",
        step=max(1, final_step // 2),
        run_id=run_ids_by_arch.get("dual_stream"),
        parent="MlpBaseline_400M",
        arch="dual_stream",
        obs_dim=58,
        manifest_hash="c4f8a91e",
        tags=["league"],
        server_url=server_url,
    )
    tl.log_checkpoint(
        "runs/saved_runs/DualStream_v2_Best.pt",
        project=project,
        name="DualStream_v2_Best",
        step=final_step,
        run_id=run_ids_by_arch.get("dual_stream"),
        parent="DualStream_v1",
        arch="dual_stream",
        obs_dim=58,
        manifest_hash="c4f8a91e",
        tags=["saved_run", "league", "best"],
        pinned=True,
        server_url=server_url,
    )

    series_results = [
        ("MlpBaseline_400M", "Random_Bot", 860, 140, 445, 500, 415, 500, "Starter_Dwarven", "Starter_Dwarven"),
        ("MlpBaseline_400M", "Heuristic_Bot", 640, 360, 338, 500, 302, 500, "Starter_Dwarven", "Starter_Heaven"),
        ("Transformer_550M", "Random_Bot", 930, 70, 475, 500, 455, 500, "Starter_Dwarven", "Starter_Dwarven"),
        ("Transformer_550M", "Heuristic_Bot", 790, 210, 412, 500, 378, 500, "Starter_Heaven", "Starter_Dwarven"),
        ("Transformer_550M", "MlpBaseline_400M", 690, 310, 362, 500, 328, 500, "Starter_Dwarven", "Starter_Dwarven"),
        ("DualStream_v1", "Heuristic_Bot", 760, 240, 395, 500, 365, 500, "Starter_Dwarven", "Starter_Heaven"),
        ("DualStream_v1", "Advanced_Bot", 590, 410, 312, 500, 278, 500, "Starter_Dwarven", "Starter_Dwarven"),
        ("DualStream_v2_Best", "Advanced_Bot", 740, 260, 385, 500, 355, 500, "Starter_Dwarven", "Starter_Heaven"),
        ("DualStream_v2_Best", "Transformer_550M", 620, 380, 324, 500, 296, 500, "Starter_Dwarven", "Starter_Dwarven"),
        ("DualStream_v2_Best", "DualStream_v1", 670, 330, 348, 500, 322, 500, "Starter_Heaven", "Starter_Dwarven"),
    ]
    for ma, mb, aw, bw, a1w, a1g, a2w, a2g, da, db_deck in series_results:
        tl.log_match_series(
            project=project,
            model_a=ma,
            model_b=mb,
            total_games=1000,
            a_wins=aw,
            b_wins=bw,
            a_as_p1_wins=a1w,
            a_as_p1_games=a1g,
            a_as_p2_wins=a2w,
            a_as_p2_games=a2g,
            deck_a=da,
            deck_b=db_deck,
            avg_rounds=6.4,
            avg_steps=38.2,
            duration_sec=1.8,
            manifest_hash="c4f8a91e",
            server_url=server_url,
        )

    # Sample 4-player OpenSkill FFA match
    tl.log_match(
        project=project,
        mode="multiplayer",
        turns=11,
        participants=[
            {"agent": "DualStream_v2_Best", "seat": 0, "deck": "Starter_Dwarven", "rank": 1, "score": 20},
            {"agent": "Transformer_550M", "seat": 1, "deck": "Starter_Heaven", "rank": 2, "score": 12},
            {"agent": "Advanced_Bot", "seat": 2, "deck": "Starter_Dwarven", "rank": 3, "score": 5},
            {"agent": "MlpBaseline_400M", "seat": 3, "deck": "Starter_Heaven", "rank": 4, "score": 0},
        ],
        server_url=server_url,
    )

    # ReDo neuron dormancy & effective rank reports (early + final for dual_stream, plus elevated dormancy on mlp)
    tl.log_plasticity(
        project=project,
        run_id=run_ids_by_arch.get("dual_stream"),
        checkpoint="DualStream_v1",
        model_name="DualStream_v1",
        step=max(1, final_step // 2),
        arch="dual_stream",
        meta={"total_obs": 4096, "hook_mode": "post_activation"},
        layers=[
            {"name": "card_encoder.0", "hook_type": "post_SiLU", "dim": 128, "hard_dead_count": 0, "hard_dead_pct": 0.0, "dormant_count": 8, "dormant_pct": 6.25, "weak_count": 19, "weak_pct": 14.8, "effective_rank": 74.2, "rank_ratio_pct": 58.0},
            {"name": "actor_head.0", "hook_type": "post_GELU", "dim": 256, "hard_dead_count": 1, "hard_dead_pct": 0.4, "dormant_count": 21, "dormant_pct": 8.2, "weak_count": 46, "weak_pct": 18.0, "effective_rank": 134.1, "rank_ratio_pct": 52.4},
            {"name": "critic_head.0", "hook_type": "Linear", "dim": 256, "hard_dead_count": 0, "hard_dead_pct": 0.0, "dormant_count": 15, "dormant_pct": 5.9, "weak_count": 32, "weak_pct": 12.5, "effective_rank": 118.0, "rank_ratio_pct": 46.1},
        ],
        server_url=server_url,
    )
    tl.log_plasticity(
        project=project,
        run_id=run_ids_by_arch.get("dual_stream"),
        checkpoint="DualStream_v2_Best",
        model_name="DualStream_v2_Best",
        step=final_step,
        arch="dual_stream",
        meta={"total_obs": 4096, "hook_mode": "post_activation"},
        layers=[
            {"name": "card_encoder.0", "hook_type": "post_SiLU", "dim": 128, "hard_dead_count": 0, "hard_dead_pct": 0.0, "dormant_count": 6, "dormant_pct": 4.7, "weak_count": 14, "weak_pct": 10.9, "effective_rank": 81.6, "rank_ratio_pct": 63.8},
            {"name": "actor_head.0", "hook_type": "post_GELU", "dim": 256, "hard_dead_count": 0, "hard_dead_pct": 0.0, "dormant_count": 18, "dormant_pct": 7.0, "weak_count": 39, "weak_pct": 15.2, "effective_rank": 141.2, "rank_ratio_pct": 55.2},
            {"name": "critic_head.0", "hook_type": "Linear", "dim": 256, "hard_dead_count": 0, "hard_dead_pct": 0.0, "dormant_count": 14, "dormant_pct": 5.5, "weak_count": 28, "weak_pct": 10.9, "effective_rank": 126.5, "rank_ratio_pct": 49.4},
        ],
        server_url=server_url,
    )
    mlp_dorm_pct = 24.5 if inject_anomaly else 14.0
    tl.log_plasticity(
        project=project,
        run_id=run_ids_by_arch.get("mlp"),
        checkpoint="MlpBaseline_400M",
        model_name="MlpBaseline_400M",
        step=final_step,
        arch="mlp",
        meta={"total_obs": 4096, "hook_mode": "post_activation"},
        layers=[
            {"name": "trunk.0", "hook_type": "post_ReLU", "dim": 512, "hard_dead_count": 42, "hard_dead_pct": 8.2, "dormant_count": int(512 * mlp_dorm_pct / 100), "dormant_pct": mlp_dorm_pct, "weak_count": 180, "weak_pct": 35.2, "effective_rank": 148.0, "rank_ratio_pct": 28.9},
            {"name": "actor.0", "hook_type": "post_ReLU", "dim": 256, "hard_dead_count": 18, "hard_dead_pct": 7.0, "dormant_count": int(256 * mlp_dorm_pct / 100), "dormant_pct": mlp_dorm_pct, "weak_count": 88, "weak_pct": 34.4, "effective_rank": 69.0, "rank_ratio_pct": 27.0},
        ],
        server_url=server_url,
    )


def _simulate_replays_and_cards(
    *,
    server_url: str,
    project: str,
    run_ids_by_arch: dict[str, str],
    final_step: int,
) -> None:
    """Upload sample Godot ReplayRecords, RL turn traces, card meta stats, and action distributions."""
    deck_dwarven = ["Iron Shieldbearer", "Rune Forge", "Anvil Golem", "Dwarven Cannon", "Firebolt"]
    deck_heaven = ["Sky Valkyrie", "Light Cleric", "Storm Caller", "Firebolt"]

    def make_replay_bytes(seed1: int, seed2: int, deck1: list[str], deck2: list[str], n_actions: int) -> bytes:
        actions = [
            {
                "type": "play_card" if i % 2 == 0 else "attack",
                "player": (i % 2) + 1,
                "hand_idx": i % 4,
                "board_idx": i % 4,
                "category": i % 3,
                "index": i,
            }
            for i in range(n_actions)
        ]
        record = {
            "version": 2,
            "manifest_hash": "c4f8a91e",
            "seed1": seed1,
            "seed2": seed2,
            "deck1": deck1,
            "deck2": deck2,
            "config": {"lanes": 4, "starting_hp": 30, "mana_cap": 10},
            "actions": actions,
        }
        return json.dumps(record, separators=(",", ":")).encode("utf-8")

    # Replay 1: DualStream vs Advanced_Bot with dramatic Turn 7 blunder
    trace_blunder = [
        {
            "turn": 1,
            "player": 1,
            "action_label": "play Iron Shieldbearer -> L1",
            "value_before": 0.50,
            "value_after": 0.53,
            "top_k_probs": [
                {"label": "play Iron Shieldbearer -> L1", "prob": 0.68, "chosen": True},
                {"label": "play Rune Forge -> L2", "prob": 0.24},
                {"label": "end_turn", "prob": 0.08},
            ],
        },
        {
            "turn": 3,
            "player": 1,
            "action_label": "play Rune Forge -> L2",
            "value_before": 0.55,
            "value_after": 0.64,
            "top_k_probs": [
                {"label": "play Rune Forge -> L2", "prob": 0.72, "chosen": True},
                {"label": "play Firebolt -> L0", "prob": 0.21},
            ],
        },
        {
            "turn": 5,
            "player": 1,
            "action_label": "play Anvil Golem -> L0",
            "value_before": 0.66,
            "value_after": 0.78,
            "top_k_probs": [
                {"label": "play Anvil Golem -> L0", "prob": 0.81, "chosen": True},
                {"label": "attack L1", "prob": 0.14},
            ],
        },
        {
            "turn": 7,
            "player": 1,
            "action_label": "end_turn (missed lethal Firebolt)",
            "value_before": 0.78,
            "value_after": 0.14,
            "top_k_probs": [
                {"label": "end_turn (missed lethal Firebolt)", "prob": 0.44, "chosen": True},
                {"label": "play Firebolt -> L3", "prob": 0.41},
                {"label": "attack L2", "prob": 0.15},
            ],
        },
        {
            "turn": 9,
            "player": 1,
            "action_label": "attack L1",
            "value_before": 0.16,
            "value_after": 0.08,
            "top_k_probs": [
                {"label": "attack L1", "prob": 0.62, "chosen": True},
                {"label": "end_turn", "prob": 0.38},
            ],
        },
    ]
    tl.log_replay(
        make_replay_bytes(918273, 445566, deck_dwarven, deck_heaven, 34),
        project=project,
        run_id=run_ids_by_arch.get("dual_stream"),
        title="DualStream vs Advanced_Bot (Turn 7 Throw)",
        filename="dualstream_vs_adv_t7_blunder.json",
        step=final_step,
        p1_name="DualStream_v2_Best",
        p2_name="Advanced_Bot",
        winner=2,
        tags=["blunder", "eval", "featured"],
        pinned=True,
        rl_trace=trace_blunder,
        server_url=server_url,
    )

    # Replay 2: Transformer clean win with trace
    trace_clean = [
        {
            "turn": t,
            "player": 1,
            "action_label": f"lane_pressure_T{t}",
            "value_before": 0.48 + 0.06 * idx,
            "value_after": 0.54 + 0.06 * idx,
            "top_k_probs": [
                {"label": f"lane_pressure_T{t}", "prob": 0.75, "chosen": True},
                {"label": "hold_mana", "prob": 0.25},
            ],
        }
        for idx, t in enumerate([1, 2, 4, 6, 8])
    ]
    tl.log_replay(
        make_replay_bytes(112233, 778899, deck_heaven, deck_dwarven, 28),
        project=project,
        run_id=run_ids_by_arch.get("transformer"),
        title="Transformer vs Heuristic (Clean Tempo)",
        filename="transformer_vs_heuristic.json",
        step=final_step,
        p1_name="Transformer_550M",
        p2_name="Heuristic_Bot",
        winner=1,
        tags=["eval", "clean"],
        rl_trace=trace_clean,
        server_url=server_url,
    )

    # Replay 3: Raw Godot ReplayRecord without trace
    tl.log_replay(
        make_replay_bytes(551100, 992200, deck_dwarven, deck_dwarven, 22),
        project=project,
        run_id=run_ids_by_arch.get("mlp"),
        title="MLP Baseline Mirror Match",
        filename="mlp_mirror_raw.json",
        step=final_step,
        p1_name="MlpBaseline_400M",
        p2_name="Random_Bot",
        winner=1,
        tags=["mirror"],
        server_url=server_url,
    )

    # Card play/draw/win statistics over 5 training checkpoints
    cards_catalog = [
        ("Firebolt", 0.62, 0.88),
        ("Iron Shieldbearer", 0.57, 0.79),
        ("Rune Forge", 0.59, 0.74),
        ("Sky Valkyrie", 0.54, 0.71),
        ("Anvil Golem", 0.56, 0.63),
        ("Storm Caller", 0.51, 0.58),
        ("Dwarven Cannon", 0.46, 0.44),
        ("Light Cleric", 0.43, 0.49),
    ]
    ds_run = run_ids_by_arch.get("dual_stream")
    steps = [max(1, int(final_step * f)) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    for s_idx, st in enumerate(steps):
        items = []
        for card_name, base_wr, play_prob in cards_catalog:
            drawn = 160 + s_idx * 35
            played = max(10, int(drawn * min(0.95, play_prob + 0.02 * s_idx)))
            wr = min(0.85, max(0.30, base_wr + 0.015 * (s_idx - 2)))
            wins_wp = int(played * wr)
            items.append(
                {
                    "card_name": card_name,
                    "played": played,
                    "drawn": drawn,
                    "wins_when_played": wins_wp,
                    "games_when_played": played,
                    "play_rate": played / drawn,
                    "win_rate": wins_wp / played,
                }
            )
        tl.log_card_stats(
            items,
            project=project,
            run_id=ds_run,
            step=st,
            server_url=server_url,
        )

    # Discrete action/lane distributions over training steps
    if ds_run:
        dist_steps = [max(1, int(final_step * f)) for f in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0)]
        for idx, st in enumerate(dist_steps):
            play_p = 0.36 + 0.03 * idx
            atk_p = 0.30 + 0.01 * idx
            abil_p = 0.14 + 0.005 * idx
            end_p = max(0.04, 1.0 - play_p - atk_p - abil_p)
            tl.log_distribution(
                "policy/action_types",
                {
                    "play_card": round(play_p, 4),
                    "attack": round(atk_p, 4),
                    "ability": round(abil_p, 4),
                    "end_turn": round(end_p, 4),
                },
                run_id=ds_run,
                step=st,
                server_url=server_url,
            )
            tl.log_distribution(
                "policy/lane_choice",
                {
                    "Lane 0": round(0.24 + 0.01 * idx, 4),
                    "Lane 1": round(0.29 - 0.005 * idx, 4),
                    "Lane 2": round(0.26, 4),
                    "Lane 3": round(max(0.10, 0.21 - 0.005 * idx), 4),
                },
                run_id=ds_run,
                step=st,
                server_url=server_url,
            )


if __name__ == "__main__":
    main()

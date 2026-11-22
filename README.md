# TrainingLog

A fast, self-hosted alternative to Weights & Biases and TensorBoard tailored for **reinforcement learning on card games**, running locally or on your personal server over LAN/VPN with a mobile-first installable PWA (warm obsidian & burnished-amber theme).

No Node.js or Bun installation is needed on any machine — the frontend uses vendored `Preact + HTM + uPlot` served directly by FastAPI.

## Features

- **Real-time Experiment Tracking & Anomaly Guard**: Lossless `int64` step series (5B+ timesteps), 4-lane cardgame presets, multi-run overlaid comparison + hyperparameter diffs, stdout/stderr log tailing, remote cooperative controls (`checkpoint`, `eval`, `run_redo`, `stop`), and mobile notifications (ntfy, Discord, Pushover, Telegram, Webhook).
- **Self-Play League, Checkpoints & ReDo Plasticity**: Exact Glicko-2 + Elo leaderboard, 4-player OpenSkill FFA ratings, P1/P2 seat bias tracking, parent→child checkpoint lineage, SQLite importer (`traininglog import-league`), and per-layer ReDo dormant-neuron / effective-rank heatmaps.
- **Godot ReplayRecord Hub & RL Blunder Traces**: Stores raw Godot replay JSON (`version`, `manifest_hash`, `seed1`, `seed2`, `deck1`, `deck2`, `actions[]`) byte-for-byte alongside MP4/WebM recordings, plus interactive turn-by-turn critic value trajectories, blunder turn jump, top-K policy probability bars, and Godot 4 `HTTPRequest` helpers.
- **Card Meta & Policy Distributions**: Aggregated per-card play/draw/win rates, temporal trends, discrete action/lane distribution evolution + entropy charts, and one-click CSV/JSON export.

## Quickstart

```bash
# Install dependencies into local venv
uv sync

# Start the server (serves API + zero-build PWA on http://0.0.0.0:8765)
uv run traininglog server --data-dir ./data --port 8765

# In another terminal, simulate 3 cardgame PPO runs + League + ReDo + Replays + Card Stats
uv run traininglog demo
```

Open `http://127.0.0.1:8765` on desktop or `http://<your-server-vpn-ip>:8765` on your phone (and tap **Add to Home Screen** to install as a PWA).

## Docker Deployment

Run TrainingLog continuously on a home server or VM over your personal VPN:

```bash
docker compose up -d --build
```

Data persists in `./data`. Set `TRAININGLOG_API_TOKEN=your-secret` in your environment before launching if you want bearer-token authentication.

## Integrating Into `cardgame/agents/ppo.py`

Replace TensorBoard `SummaryWriter` + `wandb` cleanly while keeping all existing `writer.add_scalar(...)` and `writer.add_histogram(...)` calls:

```python
import traininglog as tl

with tl.init(
    project="cardgame-ppo",
    name=run_name,
    group=args.arch,
    tags=[args.arch, args.opponent_mode],
    config=vars(args),
    arch=args.arch,
    obs_dim=CONTINUOUS_OBS_DIM,
    parent_checkpoint=checkpoint_to_load,
    total_timesteps=args.total_timesteps,
    server_url="http://<your-server-ip>:8765",  # or TRAININGLOG_URL env var
) as run:
    writer = run.summary_writer()

    for update in range(1, num_updates + 1):
        # ... existing PPO update + writer.add_scalar(...) calls ...

        # Track per-card meta statistics & discrete policy distributions
        run.log_card_stats(counts=play_counts, card_names=CARD_NAMES, step=global_step)
        run.log_distribution("policy/action_types", action_probs, labels=ACTION_LABELS, step=global_step)

        # Log Godot ReplayRecords + optional RL turn trace
        run.log_replay(replay_path, p1_name=run.name, p2_name="Advanced_Bot", rl_trace=turn_trace, step=global_step)

        # Log league results & ReDo neuron dormancy reports
        run.log_match_series(opponent="Advanced_Bot", total_games=200, wins=142, losses=58, step=global_step)
        run.log_plasticity(json_path="dormancy_report.json", step=global_step)

        # Cooperative remote controls from your phone
        for cmd in run.poll_commands():
            if cmd.type == "checkpoint":
                save_agent_checkpoint(...)
                cmd.complete("Checkpoint saved")
            elif cmd.type == "stop":
                save_agent_checkpoint(...)
                cmd.complete("Stopped cleanly")
                break
        if run.should_stop:
            break
```

Or drop in the thin `wandb` shim:

```python
import traininglog.wandb_compat as wandb
```

## Importing an Existing League SQLite DB & Godot Replay Fetching

```bash
# Import historical league.py SQLite checkpoints & matches
uv run traininglog import-league path/to/league.db --project cardgame-ppo --url http://127.0.0.1:8765
```

See [examples/godot_replay_fetcher.gd](examples/godot_replay_fetcher.gd) for a ready-to-drop Godot 4 `HTTPRequest` script to fetch raw `ReplayRecord` JSONs directly into your client for playback.

## Offline & Outage Resilience

`run.log(...)` appends to a local disk WAL spool (`~/.traininglog/spool/<run_id>/`) and batches uploads asynchronously. If the server restarts or your VPN drops, logging never blocks training and queued events flush automatically when connectivity returns.

```bash
TRAININGLOG_MODE=offline uv run python -m agents.ppo ...
uv run traininglog sync --url http://<your-server-ip>:8765
```

## Optional Token Auth

By default TrainingLog trusts localhost / LAN / personal VPN (`zero-auth`). To require a shared bearer token:

```bash
TRAININGLOG_API_TOKEN=my-secret uv run traininglog server
```

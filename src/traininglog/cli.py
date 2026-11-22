"""Command-line interface for TrainingLog (`traininglog server|sync|demo`)."""

from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from .sdk.importer import import_league_db
from .sdk.run import sync as sync_spool
from .server.app import create_app
from .server.config import ServerConfig


def _cmd_server(args: argparse.Namespace) -> int:
    cfg = ServerConfig.from_env(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        api_token=args.token,
    )
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=args.log_level)
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    res = sync_spool(
        spool_dir=args.spool_dir,
        server_url=args.url,
        api_token=args.token,
    )
    print(json.dumps(res, indent=2))
    return 1 if res.get("errors") else 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from pathlib import Path
    import runpy

    # Look for examples/simulate_cardgame_ppo.py relative to package or cwd
    candidates = [
        Path.cwd() / "examples" / "simulate_cardgame_ppo.py",
        Path(__file__).resolve().parents[2] / "examples" / "simulate_cardgame_ppo.py",
    ]
    script_path = next((p for p in candidates if p.exists()), None)
    if script_path is None:
        print("Demo simulator script not found.", file=sys.stderr)
        return 1

    saved_argv = sys.argv
    sys.argv = [
        str(script_path),
        "--url",
        args.url,
        "--updates",
        str(args.updates),
    ]
    if args.fast:
        sys.argv.append("--fast")
    if args.inject_anomaly:
        sys.argv.append("--inject-anomaly")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = saved_argv
    return 0


def _cmd_import_league(args: argparse.Namespace) -> int:
    res = import_league_db(
        args.db_path,
        project=args.project,
        server_url=args.url,
        api_token=args.token,
    )
    print(json.dumps(res, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="traininglog",
        description="Self-hosted RL experiment tracker for card-game agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_srv = sub.add_parser("server", help="Run the TrainingLog HTTP + PWA server")
    p_srv.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    p_srv.add_argument("--port", type=int, default=None, help="Bind port (default: 8765)")
    p_srv.add_argument("--data-dir", default=None, help="Storage directory (default: ./data)")
    p_srv.add_argument("--token", default=None, help="Optional API bearer token")
    p_srv.add_argument("--log-level", default="info", help="Uvicorn log level")
    p_srv.set_defaults(func=_cmd_server)

    p_sync = sub.add_parser("sync", help="Sync offline/spooled runs to the server")
    p_sync.add_argument("--spool-dir", default=None, help="Spool directory override")
    p_sync.add_argument("--url", default=None, help="Server base URL")
    p_sync.add_argument("--token", default=None, help="Optional API bearer token")
    p_sync.set_defaults(func=_cmd_sync)

    p_demo = sub.add_parser("demo", help="Simulate realistic cardgame PPO runs")
    p_demo.add_argument("--url", default="http://127.0.0.1:8765", help="Server base URL")
    p_demo.add_argument("--updates", type=int, default=40, help="PPO updates per run")
    p_demo.add_argument("--fast", action="store_true", help="Minimal sleep between updates")
    p_demo.add_argument("--inject-anomaly", action="store_true", help="Inject KL/NaN anomaly")
    p_demo.set_defaults(func=_cmd_demo)

    p_imp = sub.add_parser(
        "import-league",
        help="Import models and matches from an existing cardgame league.db",
    )
    p_imp.add_argument("db_path", help="Path to SQLite league.db")
    p_imp.add_argument("--project", default="cardgame-ppo", help="Target TrainingLog project")
    p_imp.add_argument("--url", default="http://127.0.0.1:8765", help="Server base URL")
    p_imp.add_argument("--token", default=None, help="Optional API bearer token")
    p_imp.set_defaults(func=_cmd_import_league)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

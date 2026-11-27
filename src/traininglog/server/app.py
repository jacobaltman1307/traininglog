"""FastAPI application factory, auth middleware, SSE stream, and PWA static serving."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .alerts import AlertDispatcher
from .anomalies import AnomalyDetector
from .blobs import BlobStore
from .config import ServerConfig
from .db import Database
from .meta import MetaAnalyticsEngine
from .routes import alerts as alerts_routes
from .routes import analytics as analytics_routes
from .routes import checkpoints as checkpoints_routes
from .routes import commands as commands_routes
from .routes import ingest as ingest_routes
from .routes import league as league_routes
from .routes import meta as meta_routes
from .routes import plasticity as plasticity_routes
from .routes import replays as replays_routes
from .routes import runs as runs_routes
from .series import SeriesStore
from .sse import EventBus

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    config: ServerConfig | None = None,
    *,
    alert_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = config or ServerConfig.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        db = Database(cfg.db_path)
        series_store = SeriesStore(cfg.series_dir)
        blob_store = BlobStore(cfg.blobs_dir)
        event_bus = EventBus()
        alert_dispatcher = AlertDispatcher(db, transport=alert_transport)
        anomaly_detector = AnomalyDetector(db, event_bus, alert_dispatcher)

        app.state.config = cfg
        app.state.db = db
        app.state.series_store = series_store
        app.state.blob_store = blob_store
        app.state.event_bus = event_bus
        app.state.alert_dispatcher = alert_dispatcher
        app.state.anomaly_detector = anomaly_detector
        app.state.meta_engine = MetaAnalyticsEngine(db)

        stop_event = asyncio.Event()

        async def stall_loop() -> None:
            interval = max(1, int(cfg.stall_check_interval_sec))
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    try:
                        await anomaly_detector.check_stalled_runs(
                            cfg.stall_timeout_sec
                        )
                    except Exception:
                        pass

        stall_task = asyncio.create_task(stall_loop())
        try:
            yield
        finally:
            stop_event.set()
            stall_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stall_task
            db.close()

    app = FastAPI(title="TrainingLog", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def optional_token_auth(request: Request, call_next):
        token = cfg.api_token
        path = request.url.path
        if token and path.startswith("/api/") and path != "/api/health":
            provided = None
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip()
            if not provided:
                provided = request.query_params.get("token")
            if not provided or not hmac.compare_digest(provided, token):
                return JSONResponse(
                    status_code=401, content={"detail": "Unauthorized"}
                )
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "auth_required": bool(cfg.api_token),
        }

    @app.get("/api/events")
    async def sse_events(request: Request, run_id: str | None = None):
        bus: EventBus = request.app.state.event_bus
        sub = bus.subscribe(run_id=run_id)

        async def event_generator():
            try:
                yield "event: ready\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(sub.queue.get(), timeout=15.0)
                        ev_name = item["type"]
                        payload = json.dumps(
                            item, separators=(",", ":"), allow_nan=False
                        )
                        yield f"event: {ev_name}\ndata: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                bus.unsubscribe(sub)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(ingest_routes.router)
    app.include_router(runs_routes.router)
    app.include_router(commands_routes.router)
    app.include_router(alerts_routes.router)
    app.include_router(checkpoints_routes.router)
    app.include_router(league_routes.router)
    app.include_router(plasticity_routes.router)
    app.include_router(replays_routes.router)
    app.include_router(analytics_routes.router)
    app.include_router(meta_routes.router)

    if STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )

    @app.get("/")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/manifest.webmanifest")
    def serve_manifest():
        return FileResponse(
            STATIC_DIR / "manifest.webmanifest",
            media_type="application/manifest+json",
        )

    @app.get("/sw.js")
    def serve_sw():
        return FileResponse(
            STATIC_DIR / "sw.js",
            media_type="application/javascript",
        )

    @app.get("/icon.svg")
    def serve_icon():
        return FileResponse(
            STATIC_DIR / "icon.svg",
            media_type="image/svg+xml",
        )

    return app

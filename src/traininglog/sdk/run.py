"""Core TrainingLog Run, SummaryWriter, Command, and offline spool sync."""

from __future__ import annotations

import atexit
import math
import os
import sys
import threading
import time
import traceback
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .client import SyncHttpClient
from .console import ConsoleCapture
from .provenance import collect_git_info, collect_system_info
from .spool import DiskSpool, get_default_spool_dir, iter_unsynced_run_ids
from .system_metrics import SystemMetricsCollector


def _now_ms() -> int:
    return int(time.time() * 1000)


def _serialize_scalar(val: Any) -> float | str | None:
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if hasattr(val, "item") and callable(val.item):
        try:
            val = val.item()
        except Exception:
            return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    return f


def _sanitize_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[str(k)] = None
        elif isinstance(v, dict):
            out[str(k)] = _sanitize_dict(v)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [
                None if (isinstance(x, float) and not math.isfinite(x)) else x for x in v
            ]
        elif hasattr(v, "item") and callable(v.item):
            try:
                item_v = v.item()
                if isinstance(item_v, float) and not math.isfinite(item_v):
                    out[str(k)] = None
                else:
                    out[str(k)] = item_v
            except Exception:
                out[str(k)] = str(v)
        else:
            out[str(k)] = v
    return out


class _SyncedDict:
    """Dict-like wrapper that persists config or summary updates to the spool."""

    def __init__(self, initial: dict[str, Any], on_update: Any):
        self._data: dict[str, Any] = dict(_sanitize_dict(initial))
        self._on_update = on_update
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        clean = _sanitize_dict({key: value})
        with self._lock:
            self._data.update(clean)
        self._on_update(clean)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        clean = _sanitize_dict(values)
        with self._lock:
            self._data.update(clean)
        if clean:
            self._on_update(clean)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def __repr__(self) -> str:
        with self._lock:
            return repr(self._data)


@dataclass
class Command:
    """Cooperative remote control command delivered to a training script."""

    id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    _run: Any = field(default=None, repr=False, compare=False)

    def ack(self, message: str | None = None) -> None:
        if self._run is not None:
            self._run._ack_command(self.id, status="acked", message=message)

    def complete(self, message: str | None = None) -> None:
        if self._run is not None:
            self._run._ack_command(self.id, status="completed", message=message)


class Run:
    """Active TrainingLog run handle."""

    def __init__(
        self,
        *,
        project: str = "default",
        name: str | None = None,
        run_id: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        config: dict[str, Any] | None = None,
        arch: str | None = None,
        obs_dim: int | None = None,
        manifest_hash: str | None = None,
        parent_checkpoint: str | None = None,
        total_timesteps: int | None = None,
        server_url: str | None = None,
        api_token: str | None = None,
        mode: str | None = None,
        capture_console: bool = True,
        capture_system: bool = True,
        system_interval: float = 5.0,
        command_poll_interval: float = 0.5,
        finish_timeout: float = 5.0,
        spool_dir: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.project = project
        self.name = name or self.id
        self.group = group
        self.tags = list(tags or [])
        self.notes = notes
        self.arch = arch
        self.obs_dim = obs_dim
        self.manifest_hash = manifest_hash
        self.parent_checkpoint = parent_checkpoint
        self.total_timesteps = total_timesteps
        self.mode = (mode or os.environ.get("TRAININGLOG_MODE", "online")).lower()
        self.server_url = (
            server_url or os.environ.get("TRAININGLOG_URL", "http://127.0.0.1:8765")
        ).rstrip("/")
        self.api_token = (
            api_token if api_token is not None else os.environ.get("TRAININGLOG_API_TOKEN")
        ) or None
        self._finish_timeout = float(finish_timeout)
        self._cmd_poll_interval = max(0.1, float(command_poll_interval))
        self._transport = transport

        resolved_spool_dir = Path(spool_dir) if spool_dir is not None else get_default_spool_dir()
        self._spool = DiskSpool(resolved_spool_dir, self.id)

        git_info = collect_git_info()
        sys_info = collect_system_info()
        cli_args = " ".join(sys.argv) if sys.argv else ""
        host = str(sys_info.get("hostname") or "")

        clean_config = _sanitize_dict(config or {})
        self.config = _SyncedDict(clean_config, self._on_config_update)
        self.summary = _SyncedDict({}, self._on_summary_update)

        self._step_lock = threading.Lock()
        self._last_step: int | None = None
        self._finished = False
        self._stop_worker_event = threading.Event()
        self._should_stop = False

        self._cmd_lock = threading.Lock()
        self._command_queue: list[Command] = []
        self._seen_command_ids: set[str] = set()

        self._meta_payload = {
            "run_id": self.id,
            "project": self.project,
            "name": self.name,
            "group_name": self.group,
            "tags": self.tags,
            "notes": self.notes,
            "arch": self.arch,
            "obs_dim": self.obs_dim,
            "manifest_hash": self.manifest_hash,
            "config": clean_config,
            "git": git_info,
            "system_info": sys_info,
            "cli_args": cli_args,
            "host": host,
            "parent_checkpoint_id": self.parent_checkpoint,
            "total_timesteps": self.total_timesteps,
            "started_at": _now_ms(),
        }
        self._spool.save_meta(
            {
                "server_url": self.server_url,
                "run_create_payload": self._meta_payload,
            }
        )

        self._client: SyncHttpClient | None = None
        if self.mode != "offline":
            self._client = SyncHttpClient(
                self.server_url,
                api_token=self.api_token,
                timeout=10.0,
                transport=transport,
            )

        self._console: ConsoleCapture | None = None
        if capture_console:
            self._console = ConsoleCapture(self._on_console_chunk)
            self._console.start()

        self._sys_collector: SystemMetricsCollector | None = None
        if capture_system:
            self._sys_collector = SystemMetricsCollector(
                self._on_system_metrics, interval_sec=system_interval
            )

        self._worker_thread: threading.Thread | None = None
        if self.mode != "offline":
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"traininglog-sender-{self.id[:8]}",
                daemon=True,
            )
            self._worker_thread.start()

        run_ref = weakref.ref(self)
        atexit.register(_atexit_finish, run_ref)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def _on_config_update(self, values: dict[str, Any]) -> None:
        if not self._finished:
            self._spool.append({"type": "config", "values": values})

    def _on_summary_update(self, values: dict[str, Any]) -> None:
        if not self._finished:
            self._spool.append({"type": "summary", "values": values})

    def _on_console_chunk(self, seq: int, stream: str, ts_ms: int, text: str) -> None:
        self._spool.append(
            {
                "type": "console",
                "seq": seq,
                "stream": stream,
                "ts_ms": ts_ms,
                "text": text,
            }
        )

    def _on_system_metrics(
        self, metrics: dict[str, float], gpu_snap: dict[str, Any]
    ) -> None:
        if self._finished:
            return
        if metrics:
            with self._step_lock:
                step = self._last_step if self._last_step is not None else 0
            serialized: dict[str, float | str] = {}
            for k, v in metrics.items():
                sv = _serialize_scalar(v)
                if sv is not None:
                    serialized[k] = sv
            if serialized:
                self._spool.append(
                    {
                        "type": "scalars",
                        "step": step,
                        "wall_ms": _now_ms(),
                        "metrics": serialized,
                    }
                )
        if gpu_snap:
            self._spool.append(
                {
                    "type": "heartbeat",
                    "ts_ms": _now_ms(),
                    "gpu": _sanitize_dict(gpu_snap),
                }
            )

    def _resolve_step(self, step: int | None) -> int:
        with self._step_lock:
            if step is not None:
                s = int(step)
                self._last_step = s
                return s
            if self._last_step is None:
                self._last_step = 0
            else:
                self._last_step += 1
            return self._last_step

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        wall_ms: int | None = None,
    ) -> None:
        if not metrics:
            return
        resolved_step = self._resolve_step(step)
        ts = int(wall_ms) if wall_ms is not None else _now_ms()
        serialized: dict[str, float | str] = {}
        summary_fin: dict[str, Any] = {}
        for k, raw_v in metrics.items():
            sv = _serialize_scalar(raw_v)
            if sv is None:
                continue
            serialized[str(k)] = sv
            summary_fin[str(k)] = None if isinstance(sv, str) else sv
        if not serialized:
            return
        with self.summary._lock:
            self.summary._data.update(summary_fin)
        self._spool.append(
            {
                "type": "scalars",
                "step": resolved_step,
                "wall_ms": ts,
                "metrics": serialized,
            }
        )

    def log_histogram(
        self,
        name: str,
        values: Any,
        step: int | None = None,
        bins: int = 32,
        wall_ms: int | None = None,
    ) -> None:
        if hasattr(values, "detach") and callable(values.detach):
            arr = np.asarray(values.detach().cpu().numpy(), dtype=np.float64).ravel()
        else:
            arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        resolved_step = self._resolve_step(step)
        ts = int(wall_ms) if wall_ms is not None else _now_ms()
        counts, edges = np.histogram(arr, bins=bins)
        self._spool.append(
            {
                "type": "histogram",
                "name": str(name),
                "step": resolved_step,
                "wall_ms": ts,
                "bins": edges.tolist(),
                "counts": counts.tolist(),
                "min_val": float(arr.min()),
                "max_val": float(arr.max()),
                "mean_val": float(arr.mean()),
                "std_val": float(arr.std()),
                "total_count": int(arr.size),
            }
        )

    def log_text(
        self,
        name: str,
        text: str,
        step: int | None = None,
    ) -> None:
        resolved_step = self._resolve_step(step)
        self.summary[f"text/{name}"] = text[:2000]
        if self._console is not None:
            self._console.emit_direct("stdout", f"[{name} @ step {resolved_step:,}]\n{text}")
        else:
            self._spool.append(
                {
                    "type": "console",
                    "seq": int(time.time_ns() % 2_000_000_000),
                    "stream": "stdout",
                    "ts_ms": _now_ms(),
                    "text": f"[{name} @ step {resolved_step:,}]\n{text}",
                }
            )

    def poll_commands(self) -> list[Command]:
        with self._cmd_lock:
            if not self._command_queue:
                return []
            cmds = list(self._command_queue)
            self._command_queue.clear()
            return cmds

    def _ack_command(
        self, cmd_id: str, status: str = "acked", message: str | None = None
    ) -> None:
        if self._client is None:
            return
        try:
            self._client.ack_command(
                self.id, cmd_id, status=status, result_message=message
            )
        except Exception:
            pass

    def summary_writer(self) -> SummaryWriter:
        return SummaryWriter(run=self)

    def _ensure_registered_for_direct_api(self) -> None:
        if self.mode != "offline" and not self._spool.is_registered():
            self.flush(timeout=2.0)

    def log_checkpoint(
        self,
        path_or_name: str | Path,
        *,
        name: str | None = None,
        step: int | None = None,
        parent: str | None = None,
        arch: str | None = None,
        obs_dim: int | None = None,
        manifest_hash: str | None = None,
        tags: list[str] | None = None,
        pinned: bool = False,
        metrics: dict[str, Any] | None = None,
        upload: bool = False,
        compute_hash: bool = False,
    ) -> dict[str, Any]:
        from .league import log_checkpoint as _log_checkpoint

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else int(self._last_step or 0)
        return _log_checkpoint(
            path_or_name,
            project=self.project,
            name=name,
            step=resolved_step,
            run_id=self.id,
            parent=parent or self.parent_checkpoint,
            arch=arch or self.arch,
            obs_dim=obs_dim if obs_dim is not None else self.obs_dim,
            manifest_hash=manifest_hash or self.manifest_hash,
            tags=tags,
            pinned=pinned,
            metrics=metrics,
            upload=upload,
            compute_hash=compute_hash,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_match_series(
        self,
        *,
        model_a: str,
        model_b: str,
        total_games: int,
        a_wins: int,
        b_wins: int,
        draws: int = 0,
        a_as_p1_wins: int = 0,
        a_as_p1_games: int = 0,
        a_as_p2_wins: int = 0,
        a_as_p2_games: int = 0,
        b_as_p1_wins: int | None = None,
        b_as_p1_games: int | None = None,
        b_as_p2_wins: int | None = None,
        b_as_p2_games: int | None = None,
        deck_a: str | None = None,
        deck_b: str | None = None,
        avg_rounds: float = 0.0,
        avg_steps: float = 0.0,
        duration_sec: float = 0.0,
        step: int | None = None,
        manifest_hash: str | None = None,
        source_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .league import log_match_series as _log_match_series

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_match_series(
            project=self.project,
            model_a=model_a,
            model_b=model_b,
            total_games=total_games,
            a_wins=a_wins,
            b_wins=b_wins,
            draws=draws,
            a_as_p1_wins=a_as_p1_wins,
            a_as_p1_games=a_as_p1_games,
            a_as_p2_wins=a_as_p2_wins,
            a_as_p2_games=a_as_p2_games,
            b_as_p1_wins=b_as_p1_wins,
            b_as_p1_games=b_as_p1_games,
            b_as_p2_wins=b_as_p2_wins,
            b_as_p2_games=b_as_p2_games,
            deck_a=deck_a,
            deck_b=deck_b,
            avg_rounds=avg_rounds,
            avg_steps=avg_steps,
            duration_sec=duration_sec,
            step=resolved_step,
            run_id=self.id,
            manifest_hash=manifest_hash or self.manifest_hash,
            source_key=source_key,
            metadata=metadata,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_match(
        self,
        *,
        participants: list[dict[str, Any]],
        mode: str | None = None,
        turns: int = 0,
        duration_sec: float = 0.0,
        step: int | None = None,
        manifest_hash: str | None = None,
        source_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .league import log_match as _log_match

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_match(
            project=self.project,
            participants=participants,
            mode=mode,
            turns=turns,
            duration_sec=duration_sec,
            step=resolved_step,
            run_id=self.id,
            manifest_hash=manifest_hash or self.manifest_hash,
            source_key=source_key,
            metadata=metadata,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_rating(
        self,
        *,
        agent: str,
        glicko_mu: float | None = None,
        glicko_rd: float | None = None,
        elo: float | None = None,
        os_mu: float | None = None,
        os_sigma: float | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        from .league import log_rating as _log_rating

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_rating(
            project=self.project,
            agent=agent,
            glicko_mu=glicko_mu,
            glicko_rd=glicko_rd,
            elo=elo,
            os_mu=os_mu,
            os_sigma=os_sigma,
            step=resolved_step,
            run_id=self.id,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_plasticity(
        self,
        *,
        layers: list[Any] | None = None,
        meta: dict[str, Any] | None = None,
        json_path: str | Path | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        step: int | None = None,
        arch: str | None = None,
        hook_mode: str | None = None,
    ) -> dict[str, Any]:
        from .league import log_plasticity as _log_plasticity

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_plasticity(
            project=self.project,
            layers=layers,
            meta=meta,
            json_path=json_path,
            model_name=model_name or self.name,
            checkpoint=checkpoint,
            run_id=self.id,
            step=resolved_step,
            arch=arch or self.arch,
            hook_mode=hook_mode,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_replay(
        self,
        path_or_bytes: str | Path | bytes,
        *,
        match_id: str | None = None,
        title: str | None = None,
        format: str | None = None,
        step: int | None = None,
        p1_name: str = "",
        p2_name: str = "",
        winner: int | None = None,
        tags: list[str] | None = None,
        pinned: bool = False,
        rl_trace: list[dict[str, Any]] | dict[str, Any] | str | Path | None = None,
        meta: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        from .replays import log_replay as _log_replay

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_replay(
            path_or_bytes,
            project=self.project,
            run_id=self.id,
            match_id=match_id,
            title=title,
            format=format,
            step=resolved_step,
            p1_name=p1_name,
            p2_name=p2_name,
            winner=winner,
            tags=tags,
            pinned=pinned,
            rl_trace=rl_trace,
            meta=meta,
            filename=filename,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_card_stats(
        self,
        cards: list[dict[str, Any]] | dict[str, Any] | None = None,
        *,
        counts: Any = None,
        card_names: list[str] | None = None,
        drawn: Any = None,
        wins_when_played: Any = None,
        games_when_played: Any = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        from .replays import log_card_stats as _log_card_stats

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_card_stats(
            cards,
            counts=counts,
            card_names=card_names,
            drawn=drawn,
            wins_when_played=wins_when_played,
            games_when_played=games_when_played,
            project=self.project,
            run_id=self.id,
            step=resolved_step,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def log_distribution(
        self,
        name: str,
        probs: list[float] | dict[str, float] | Any,
        *,
        labels: list[str] | None = None,
        step: int | None = None,
        entropy: float | None = None,
    ) -> dict[str, Any]:
        from .replays import log_distribution as _log_distribution

        self._ensure_registered_for_direct_api()
        resolved_step = int(step) if step is not None else self._last_step
        return _log_distribution(
            name,
            probs,
            labels=labels,
            run_id=self.id,
            step=resolved_step,
            entropy=entropy,
            server_url=self.server_url,
            api_token=self.api_token,
            transport=self._transport,
        )

    def _worker_loop(self) -> None:
        assert self._client is not None
        failures = 0
        last_cmd_poll = 0.0
        last_heartbeat = time.monotonic()

        while not self._stop_worker_event.is_set():
            did_work = False
            try:
                if not self._spool.is_registered():
                    self._client.create_run(self._meta_payload)
                    self._spool.mark_registered(True)
                    failures = 0
                    did_work = True

                while not self._stop_worker_event.is_set():
                    events, max_seq, new_offset = self._spool.read_pending(limit=200)
                    if not events:
                        break
                    self._client.post_events(self.id, max_seq, events)
                    self._spool.mark_acked(max_seq, new_offset)
                    failures = 0
                    did_work = True
                    last_heartbeat = time.monotonic()

                now_mono = time.monotonic()
                if (now_mono - last_cmd_poll) >= self._cmd_poll_interval:
                    last_cmd_poll = now_mono
                    cmds = self._client.get_pending_commands(self.id)
                    if cmds:
                        with self._cmd_lock:
                            for raw in cmds:
                                cid = raw["id"]
                                if cid in self._seen_command_ids:
                                    continue
                                self._seen_command_ids.add(cid)
                                cmd_obj = Command(
                                    id=cid,
                                    type=raw["cmd_type"],
                                    payload=raw.get("payload") or {},
                                    created_at=int(raw.get("created_at") or 0),
                                    _run=self,
                                )
                                self._command_queue.append(cmd_obj)
                                if cmd_obj.type == "stop":
                                    self._should_stop = True

                if (now_mono - last_heartbeat) >= 15.0 and not self._finished:
                    last_heartbeat = now_mono
                    self._spool.append({"type": "heartbeat", "ts_ms": _now_ms()})

                pending_finish = self._spool.get_pending_finish()
                if pending_finish is not None and self._spool.pending_event_count() == 0:
                    self._client.finish_run(self.id, pending_finish)
                    self._spool.mark_finish_sent()
                    failures = 0
                    break

            except Exception:
                failures += 1
                backoff = min(5.0, 0.2 * (2 ** min(failures - 1, 5)))
                with self._spool.notify_cond:
                    self._spool.notify_cond.wait(timeout=backoff)
                continue

            if not did_work:
                wait_s = min(0.25, self._cmd_poll_interval)
                with self._spool.notify_cond:
                    if not self._spool.has_pending_work() and not self._stop_worker_event.is_set():
                        self._spool.notify_cond.wait(timeout=wait_s)

    def flush(self, timeout: float = 10.0) -> bool:
        if self._console is not None:
            self._console.flush()
        if self.mode == "offline":
            return True
        deadline = time.monotonic() + float(timeout)
        with self._spool.notify_cond:
            while time.monotonic() < deadline:
                if (
                    self._spool.is_registered()
                    and self._spool.pending_event_count() == 0
                ):
                    return True
                rem = max(0.01, deadline - time.monotonic())
                self._spool.notify_cond.wait(timeout=min(0.1, rem))
            return (
                self._spool.is_registered()
                and self._spool.pending_event_count() == 0
            )

    def finish(
        self,
        exit_code: int = 0,
        state: str | None = None,
        error_message: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True

        if self._sys_collector is not None:
            self._sys_collector.stop()
        if self._console is not None:
            self._console.stop()

        final_state = state or ("crashed" if exit_code != 0 else "finished")
        finish_payload = {
            "state": final_state,
            "summary": self.summary.to_dict(),
            "finished_at": _now_ms(),
            "error_message": error_message,
        }
        self._spool.set_pending_finish(finish_payload)

        if self._worker_thread is not None:
            wait_sec = self._finish_timeout if timeout is None else float(timeout)
            deadline = time.monotonic() + wait_sec
            with self._spool.notify_cond:
                while time.monotonic() < deadline:
                    if not self._spool.has_pending_work():
                        break
                    rem = max(0.01, deadline - time.monotonic())
                    self._spool.notify_cond.wait(timeout=min(0.1, rem))
            self._stop_worker_event.set()
            with self._spool.notify_cond:
                self._spool.notify_cond.notify_all()
            self._worker_thread.join(timeout=1.5)

        if self._client is not None:
            self._client.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            if self._console is not None:
                self._console.emit_direct("stderr", tb_text)
            else:
                self._spool.append(
                    {
                        "type": "console",
                        "seq": int(time.time_ns() % 2_000_000_000),
                        "stream": "stderr",
                        "ts_ms": _now_ms(),
                        "text": tb_text.rstrip("\r\n"),
                    }
                )
            self.finish(
                exit_code=1,
                state="crashed",
                error_message=f"{exc_type.__name__}: {exc_val}",
            )
            return False
        self.finish(exit_code=0, state="finished")
        return False


def _atexit_finish(run_ref: weakref.ref[Run]) -> None:
    run = run_ref()
    if run is not None and not run._finished:
        try:
            run.finish(exit_code=0, timeout=2.0)
        except Exception:
            pass


def init(
    project: str = "default",
    *,
    name: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    config: dict[str, Any] | None = None,
    arch: str | None = None,
    obs_dim: int | None = None,
    manifest_hash: str | None = None,
    parent_checkpoint: str | None = None,
    total_timesteps: int | None = None,
    server_url: str | None = None,
    api_token: str | None = None,
    mode: str | None = None,
    capture_console: bool = True,
    capture_system: bool = True,
    system_interval: float = 5.0,
    command_poll_interval: float = 0.5,
    finish_timeout: float = 5.0,
    spool_dir: str | Path | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Run:
    """Create and start a TrainingLog Run."""
    return Run(
        project=project,
        name=name,
        run_id=run_id,
        group=group,
        tags=tags,
        notes=notes,
        config=config,
        arch=arch,
        obs_dim=obs_dim,
        manifest_hash=manifest_hash,
        parent_checkpoint=parent_checkpoint,
        total_timesteps=total_timesteps,
        server_url=server_url,
        api_token=api_token,
        mode=mode,
        capture_console=capture_console,
        capture_system=capture_system,
        system_interval=system_interval,
        command_poll_interval=command_poll_interval,
        finish_timeout=finish_timeout,
        spool_dir=spool_dir,
        transport=transport,
    )


class SummaryWriter:
    """Drop-in replacement for torch.utils.tensorboard.SummaryWriter."""

    def __init__(
        self,
        log_dir: str | None = None,
        *,
        run: Run | None = None,
        project: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        server_url: str | None = None,
        api_token: str | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ):
        self._owns_run = run is None
        if run is not None:
            self._run = run
        else:
            inferred_name = name or (Path(log_dir).name if log_dir else None)
            self._run = init(
                project=project or "default",
                name=inferred_name,
                config=config,
                server_url=server_url,
                api_token=api_token,
                mode=mode,
            )

    @property
    def run(self) -> Run:
        return self._run

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        wall_ms = int(walltime * 1000) if walltime is not None else None
        self._run.log({tag: scalar_value}, step=global_step, wall_ms=wall_ms)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, Any],
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        wall_ms = int(walltime * 1000) if walltime is not None else None
        prefixed = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self._run.log(prefixed, step=global_step, wall_ms=wall_ms)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        bins: int | str = 32,
        walltime: float | None = None,
        max_bins: int | None = None,
    ) -> None:
        wall_ms = int(walltime * 1000) if walltime is not None else None
        n_bins = max_bins or (32 if isinstance(bins, str) else int(bins))
        self._run.log_histogram(
            tag, values, step=global_step, bins=n_bins, wall_ms=wall_ms
        )

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        self._run.log_text(tag, text_string, step=global_step)

    def flush(self) -> None:
        self._run.flush()

    def close(self) -> None:
        if self._owns_run:
            self._run.finish()
        else:
            self._run.flush()


def sync(
    spool_dir: str | Path | None = None,
    *,
    server_url: str | None = None,
    api_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Upload all pending offline/unsynced runs from local WAL spool to the server."""
    root = Path(spool_dir) if spool_dir is not None else get_default_spool_dir()
    run_ids = iter_unsynced_run_ids(root)
    synced_runs: list[str] = []
    synced_events = 0
    errors: list[dict[str, str]] = []

    token = (
        api_token if api_token is not None else os.environ.get("TRAININGLOG_API_TOKEN")
    ) or None

    for rid in run_ids:
        spool = DiskSpool(root, rid)
        meta = spool.load_meta()
        target_url = (
            server_url
            or meta.get("server_url")
            or os.environ.get("TRAININGLOG_URL", "http://127.0.0.1:8765")
        ).rstrip("/")
        client = SyncHttpClient(
            target_url, api_token=token, timeout=15.0, transport=transport
        )
        try:
            if not spool.is_registered():
                create_payload = meta.get("run_create_payload") or {"run_id": rid}
                client.create_run(create_payload)
                spool.mark_registered(True)

            while True:
                events, max_seq, new_offset = spool.read_pending(limit=500)
                if not events:
                    break
                client.post_events(rid, max_seq, events)
                spool.mark_acked(max_seq, new_offset)
                synced_events += len(events)

            pending_finish = spool.get_pending_finish()
            if pending_finish is not None and spool.pending_event_count() == 0:
                client.finish_run(rid, pending_finish)
                spool.mark_finish_sent()

            synced_runs.append(rid)
        except Exception as exc:
            errors.append({"run_id": rid, "error": str(exc)})
        finally:
            client.close()

    return {
        "synced_runs": synced_runs,
        "synced_events": synced_events,
        "errors": errors,
    }

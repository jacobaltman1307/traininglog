"""Local disk Write-Ahead Log (WAL) spool for zero-drop metric & console delivery."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def get_default_spool_dir() -> Path:
    env_dir = os.environ.get("TRAININGLOG_SPOOL_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".traininglog" / "spool"


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class DiskSpool:
    """Thread-safe per-run WAL queue storing events safely on disk."""

    def __init__(self, spool_dir: Path, run_id: str):
        self.spool_dir = Path(spool_dir)
        self.run_id = run_id
        self.run_dir = self.spool_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.run_dir / "meta.json"
        self.wal_path = self.run_dir / "events.wal"
        self.cursor_path = self.run_dir / "cursor.json"

        self._lock = threading.RLock()
        self.notify_cond = threading.Condition(self._lock)

        self._cursor = _read_json(
            self.cursor_path,
            {
                "acked_seq": 0,
                "acked_offset": 0,
                "run_registered": False,
                "finish_sent": False,
                "pending_finish": None,
            },
        )
        self._next_seq = self._discover_next_seq()

    def _discover_next_seq(self) -> int:
        max_seq = int(self._cursor.get("acked_seq", 0))
        if self.wal_path.exists():
            try:
                with open(self.wal_path, "rb") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            seq = int(obj.get("event_seq", 0))
                            if seq > max_seq:
                                max_seq = seq
                        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                            continue
            except OSError:
                pass
        return max_seq + 1

    def save_meta(self, meta: dict[str, Any]) -> None:
        with self._lock:
            _atomic_write_json(self.meta_path, meta)
            self.notify_cond.notify_all()

    def load_meta(self) -> dict[str, Any]:
        with self._lock:
            return _read_json(self.meta_path, {})

    def append(self, event: dict[str, Any]) -> int:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            frame = dict(event)
            frame["event_seq"] = seq
            encoded = (
                json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            with open(self.wal_path, "ab") as f:
                f.write(encoded)
                f.flush()
            self.notify_cond.notify_all()
            return seq

    def read_pending(self, limit: int = 500) -> tuple[list[dict[str, Any]], int, int]:
        """Return (events, max_seq_in_batch, file_offset_after_batch)."""
        with self._lock:
            if not self.wal_path.exists():
                return [], int(self._cursor.get("acked_seq", 0)), 0

            acked_seq = int(self._cursor.get("acked_seq", 0))
            offset = int(self._cursor.get("acked_offset", 0))
            file_size = self.wal_path.stat().st_size
            if offset < 0 or offset > file_size:
                offset = 0

            events: list[dict[str, Any]] = []
            max_seq = acked_seq
            new_offset = offset

            with open(self.wal_path, "rb") as f:
                f.seek(offset)
                while len(events) < limit:
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # Partial trailing write safety
                        break
                    new_offset = f.tell()
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    seq = int(obj.get("event_seq", 0))
                    if seq <= acked_seq:
                        continue
                    events.append(obj)
                    if seq > max_seq:
                        max_seq = seq

            return events, max_seq, new_offset

    def mark_acked(self, up_to_seq: int, new_offset: int | None = None) -> None:
        with self._lock:
            if up_to_seq > int(self._cursor.get("acked_seq", 0)):
                self._cursor["acked_seq"] = int(up_to_seq)
            if new_offset is not None and new_offset > int(
                self._cursor.get("acked_offset", 0)
            ):
                self._cursor["acked_offset"] = int(new_offset)
            _atomic_write_json(self.cursor_path, self._cursor)
            self.notify_cond.notify_all()

    def is_registered(self) -> bool:
        with self._lock:
            return bool(self._cursor.get("run_registered", False))

    def mark_registered(self, flag: bool = True) -> None:
        with self._lock:
            self._cursor["run_registered"] = flag
            _atomic_write_json(self.cursor_path, self._cursor)
            self.notify_cond.notify_all()

    def set_pending_finish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._cursor["pending_finish"] = payload
            self._cursor["finish_sent"] = False
            _atomic_write_json(self.cursor_path, self._cursor)
            self.notify_cond.notify_all()

    def get_pending_finish(self) -> dict[str, Any] | None:
        with self._lock:
            if self._cursor.get("finish_sent"):
                return None
            return self._cursor.get("pending_finish")

    def mark_finish_sent(self) -> None:
        with self._lock:
            self._cursor["finish_sent"] = True
            _atomic_write_json(self.cursor_path, self._cursor)
            self.notify_cond.notify_all()

    def has_pending_work(self) -> bool:
        with self._lock:
            if not self._cursor.get("run_registered"):
                return True
            if self._cursor.get("pending_finish") and not self._cursor.get("finish_sent"):
                return True
            acked_seq = int(self._cursor.get("acked_seq", 0))
            return (self._next_seq - 1) > acked_seq

    def pending_event_count(self) -> int:
        with self._lock:
            acked_seq = int(self._cursor.get("acked_seq", 0))
            return max(0, (self._next_seq - 1) - acked_seq)


def iter_unsynced_run_ids(base_spool_dir: Path | None = None) -> list[str]:
    root = Path(base_spool_dir) if base_spool_dir is not None else get_default_spool_dir()
    if not root.exists():
        return []
    run_ids: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "meta.json").exists():
            continue
        spool = DiskSpool(root, child.name)
        if spool.has_pending_work():
            run_ids.append(child.name)
    return run_ids

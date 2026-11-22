"""Binary columnar time-series store (.tlts) with vectorized M4 downsampling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np

RECORD_DTYPE = np.dtype([("step", "<i8"), ("wall_ms", "<i8"), ("val", "<f8")])
RECORD_SIZE = RECORD_DTYPE.itemsize  # 24 bytes
MAX_NON_FINITE_REPORTED = 60


def metric_filename(metric_name: str) -> str:
    digest = hashlib.sha1(metric_name.encode("utf-8")).hexdigest()[:16]
    return f"m_{digest}.tlts"


def _parse_float(value: Any) -> float:
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "nan":
            return float("nan")
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if low in ("-inf", "-infinity"):
            return float("-inf")
    return float(value)


def _finite_or_none(val: float | None) -> float | None:
    if val is None or not math.isfinite(val):
        return None
    return float(val)


def _m4_downsample_indices(vals: np.ndarray, max_points: int) -> np.ndarray:
    """Vectorized/fast M4 (First, Min, Max, Last per bucket) downsampling preserving spikes & NaNs."""
    n = len(vals)
    if n <= max_points or max_points < 8:
        return np.arange(n, dtype=np.int64)

    finite_mask = np.isfinite(vals)
    non_fin = np.flatnonzero(~finite_mask) if not finite_mask.all() else np.empty(0, dtype=np.int64)
    extra_nan_budget = min(len(non_fin), MAX_NON_FINITE_REPORTED) if len(non_fin) > 0 else 0
    bucket_budget = max(8, max_points - extra_nan_budget) if max_points > extra_nan_budget + 8 else max_points
    n_buckets = max(1, bucket_budget // 4)
    edges = np.linspace(0, n, n_buckets + 1, dtype=np.int64)

    picks: list[int] = [0, n - 1]
    for b in range(n_buckets):
        s = int(edges[b])
        e = int(edges[b + 1])
        if s >= e:
            continue
        picks.append(s)
        picks.append(e - 1)
        seg_fin = finite_mask[s:e]
        if seg_fin.all():
            seg = vals[s:e]
            picks.append(s + int(np.argmin(seg)))
            picks.append(s + int(np.argmax(seg)))
        elif seg_fin.any():
            fin_positions = np.flatnonzero(seg_fin)
            fin_vals = vals[s + fin_positions]
            picks.append(s + int(fin_positions[int(np.argmin(fin_vals))]))
            picks.append(s + int(fin_positions[int(np.argmax(fin_vals))]))

    if len(non_fin) > 0:
        if len(non_fin) <= MAX_NON_FINITE_REPORTED:
            picks.extend(non_fin.tolist())
        else:
            picks.extend(non_fin[:50].tolist())
            picks.extend(non_fin[-10:].tolist())

    return np.unique(np.asarray(picks, dtype=np.int64))


class SeriesStore:
    """Per-run binary time-series store backed by .tlts files and atomic index.json."""

    def __init__(self, series_dir: Path):
        self.series_dir = Path(series_dir)
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}
        self._index_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def _get_lock(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.Lock()
                self._run_locks[run_id] = lock
            return lock

    def _run_dir(self, run_id: str) -> Path:
        return self.series_dir / run_id

    def _load_index(self, run_id: str) -> dict[str, dict[str, Any]]:
        if run_id in self._index_cache:
            return self._index_cache[run_id]
        idx_path = self._run_dir(run_id) / "index.json"
        if idx_path.exists():
            try:
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                self._index_cache[run_id] = data
                return data
            except (OSError, json.JSONDecodeError):
                pass
        data: dict[str, dict[str, Any]] = {}
        self._index_cache[run_id] = data
        return data

    def _save_index(self, run_id: str, index: dict[str, dict[str, Any]]) -> None:
        rdir = self._run_dir(run_id)
        rdir.mkdir(parents=True, exist_ok=True)
        idx_path = rdir / "index.json"
        tmp_path = rdir / "index.json.tmp"
        tmp_path.write_text(
            json.dumps(index, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, idx_path)

    def append_points(
        self, run_id: str, points_by_metric: dict[str, list[tuple[int, int, Any]]]
    ) -> dict[str, dict[str, Any]]:
        """Append (step, wall_ms, val) points per metric. Returns updated metric index entries."""
        if not points_by_metric:
            return {}
        lock = self._get_lock(run_id)
        updated: dict[str, dict[str, Any]] = {}
        with lock:
            rdir = self._run_dir(run_id)
            rdir.mkdir(parents=True, exist_ok=True)
            index = self._load_index(run_id)

            for metric, pts in points_by_metric.items():
                if not pts:
                    continue
                entry = index.get(metric)
                if entry is None:
                    fname = metric_filename(metric)
                    entry = {
                        "name": metric,
                        "file": fname,
                        "count": 0,
                        "min": None,
                        "max": None,
                        "last_step": None,
                        "last_wall_ms": None,
                        "last_val": None,
                        "has_nan": False,
                    }
                    index[metric] = entry

                parsed_rows: list[tuple[int, int, float]] = []
                for step, wall_ms, raw_val in pts:
                    val = _parse_float(raw_val)
                    parsed_rows.append((int(step), int(wall_ms), val))
                    entry["count"] += 1
                    entry["last_step"] = int(step)
                    entry["last_wall_ms"] = int(wall_ms)
                    if math.isfinite(val):
                        entry["last_val"] = float(val)
                        if entry["min"] is None or val < entry["min"]:
                            entry["min"] = float(val)
                        if entry["max"] is None or val > entry["max"]:
                            entry["max"] = float(val)
                    else:
                        entry["has_nan"] = True
                        entry["last_val"] = None

                arr = np.asarray(parsed_rows, dtype=RECORD_DTYPE)
                fpath = rdir / entry["file"]
                with open(fpath, "ab") as f:
                    f.write(arr.tobytes())

                updated[metric] = dict(entry)

            self._save_index(run_id, index)
        return updated

    def list_metrics(self, run_id: str) -> list[dict[str, Any]]:
        lock = self._get_lock(run_id)
        with lock:
            index = self._load_index(run_id)
            return sorted(
                (dict(v) for v in index.values()),
                key=lambda item: item["name"],
            )

    def _read_records(self, run_id: str, filename: str) -> np.ndarray:
        fpath = self._run_dir(run_id) / filename
        if not fpath.exists():
            return np.empty(0, dtype=RECORD_DTYPE)
        size = fpath.stat().st_size
        usable = (size // RECORD_SIZE) * RECORD_SIZE
        if usable <= 0:
            return np.empty(0, dtype=RECORD_DTYPE)
        with open(fpath, "rb") as f:
            raw = f.read(usable)
        return np.frombuffer(raw, dtype=RECORD_DTYPE)

    def query_series(
        self,
        run_id: str,
        metric: str,
        *,
        max_points: int = 800,
        x_axis: str = "step",
        run_started_ms: int = 0,
        min_step: int | None = None,
        max_step: int | None = None,
    ) -> dict[str, Any]:
        lock = self._get_lock(run_id)
        with lock:
            index = self._load_index(run_id)
            entry = index.get(metric)
            if entry is None:
                return {
                    "metric": metric,
                    "total_points": 0,
                    "returned_points": 0,
                    "x": [],
                    "y": [],
                    "non_finite_steps": [],
                }
            records = self._read_records(run_id, entry["file"])

        if len(records) == 0:
            return {
                "metric": metric,
                "total_points": 0,
                "returned_points": 0,
                "x": [],
                "y": [],
                "non_finite_steps": [],
            }

        if min_step is not None or max_step is not None:
            mask = np.ones(len(records), dtype=bool)
            if min_step is not None:
                mask &= records["step"] >= int(min_step)
            if max_step is not None:
                mask &= records["step"] <= int(max_step)
            records = records[mask]

        total_points = int(len(records))
        if total_points == 0:
            return {
                "metric": metric,
                "total_points": 0,
                "returned_points": 0,
                "x": [],
                "y": [],
                "non_finite_steps": [],
            }

        steps = records["step"]
        vals = records["val"]
        non_fin_mask = ~np.isfinite(vals)
        if non_fin_mask.any():
            bad_steps = steps[non_fin_mask]
            if len(bad_steps) > MAX_NON_FINITE_REPORTED:
                non_finite_steps = (
                    bad_steps[:50].tolist() + bad_steps[-10:].tolist()
                )
            else:
                non_finite_steps = bad_steps.tolist()
        else:
            non_finite_steps = []

        sel_idx = _m4_downsample_indices(vals, max_points)
        sel_records = records[sel_idx]

        if x_axis == "wall_time":
            x_list = (sel_records["wall_ms"].astype(np.float64) / 1000.0).tolist()
        elif x_axis == "relative_time":
            x_list = (
                (sel_records["wall_ms"] - int(run_started_ms)).astype(np.float64)
                / 1000.0
            ).tolist()
        else:
            x_list = sel_records["step"].tolist()

        sel_vals = sel_records["val"]
        if np.isfinite(sel_vals).all():
            y_list: list[float | None] = sel_vals.tolist()
        else:
            y_list = [
                float(v) if math.isfinite(v) else None for v in sel_vals.tolist()
            ]

        return {
            "metric": metric,
            "total_points": total_points,
            "returned_points": len(x_list),
            "x": x_list,
            "y": y_list,
            "non_finite_steps": non_finite_steps,
        }

    def export_run_rows(self, run_id: str) -> list[dict[str, Any]]:
        """Read all scalar records for a run ordered by step then metric."""
        lock = self._get_lock(run_id)
        with lock:
            index = self._load_index(run_id)
            entries = sorted(index.values(), key=lambda e: e["name"])
            all_rows: list[tuple[int, int, str, float | None]] = []
            for entry in entries:
                recs = self._read_records(run_id, entry["file"])
                mname = entry["name"]
                for step, wall_ms, val in recs.tolist():
                    all_rows.append(
                        (int(step), int(wall_ms), mname, _finite_or_none(float(val)))
                    )
        all_rows.sort(key=lambda r: (r[0], r[1], r[2]))
        return [
            {"step": s, "wall_ms": w, "metric": m, "value": v}
            for s, w, m, v in all_rows
        ]

    def delete_run(self, run_id: str) -> None:
        lock = self._get_lock(run_id)
        with lock:
            self._index_cache.pop(run_id, None)
            rdir = self._run_dir(run_id)
            if rdir.exists():
                shutil.rmtree(rdir, ignore_errors=True)
        with self._locks_guard:
            self._run_locks.pop(run_id, None)

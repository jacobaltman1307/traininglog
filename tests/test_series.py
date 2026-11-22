"""Tests for binary .tlts time-series store, int64 5B+ steps, M4 downsampling, and strict JSON."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from traininglog.server.series import SeriesStore


def test_5b_int64_steps_and_x_axes(tmp_path: Path) -> None:
    store = SeriesStore(tmp_path / "series")
    pts = [
        (5_000_000_000, 1_700_000_000_000, 0.42),
        (10_000_000_000, 1_700_000_060_000, 0.78),
    ]
    store.append_points("run_1", {"game/win_rate_overall": pts})

    res_step = store.query_series("run_1", "game/win_rate_overall", x_axis="step")
    assert res_step["x"] == [5_000_000_000, 10_000_000_000]
    assert res_step["y"] == [0.42, 0.78]

    res_rel = store.query_series(
        "run_1",
        "game/win_rate_overall",
        x_axis="relative_time",
        run_started_ms=1_700_000_000_000,
    )
    assert res_rel["x"] == [0.0, 60.0]

    res_filtered = store.query_series(
        "run_1",
        "game/win_rate_overall",
        min_step=6_000_000_000,
    )
    assert res_filtered["x"] == [10_000_000_000]
    assert res_filtered["y"] == [0.78]


def test_500k_m4_downsampling_preserves_spikes_and_performance(tmp_path: Path) -> None:
    store = SeriesStore(tmp_path / "series")
    n = 500_000
    base_ms = 1_700_000_000_000
    pts = []
    for i in range(n):
        val = math.sin(i * 0.01) * 0.5
        if i == 123_456:
            val = 9999.0  # single-step KL spike
        elif i == 345_678:
            val = -5555.0  # single-step dip
        pts.append((i * 1000, base_ms + i, val))

    store.append_points("run_500k", {"losses/approx_kl": pts})

    t0 = time.perf_counter()
    res = store.query_series("run_500k", "losses/approx_kl", max_points=800)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 200.0, f"Downsampling 500k points took {elapsed_ms:.1f}ms"
    assert res["total_points"] == n
    assert res["returned_points"] <= 800
    assert res["x"][0] == 0
    assert res["x"][-1] == (n - 1) * 1000
    assert 9999.0 in res["y"]
    assert -5555.0 in res["y"]


def test_non_finite_values_produce_strict_json(tmp_path: Path) -> None:
    store = SeriesStore(tmp_path / "series")
    pts = [
        (10, 1000, 1.0),
        (20, 2000, "NaN"),
        (30, 3000, "Infinity"),
        (40, 4000, "-Infinity"),
        (50, 5000, 2.5),
    ]
    updated = store.append_points("run_nan", {"losses/value_loss": pts})
    assert updated["losses/value_loss"]["has_nan"] is True
    assert updated["losses/value_loss"]["min"] == 1.0
    assert updated["losses/value_loss"]["max"] == 2.5

    res = store.query_series("run_nan", "losses/value_loss", max_points=800)
    assert res["non_finite_steps"] == [20, 30, 40]
    assert res["y"] == [1.0, None, None, None, 2.5]

    encoded = json.dumps(res, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["y"] == [1.0, None, None, None, 2.5]

    rows = store.export_run_rows("run_nan")
    assert len(rows) == 5
    json.dumps(rows, allow_nan=False)
